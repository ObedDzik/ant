"""
Test-time adaptation evaluation for prostate cancer detection (Gilany et al., 2024).

Implements:
  - Baseline: no adaptation, just forward pass
  - TENT: entropy minimization on GroupNorm params (Wang et al., 2021)
  - TTT: test-time training with BYOL self-supervised head (Sun et al., 2020)
  - DEnEM: diverse ensemble entropy minimization (Gilany et al., 2024)

Evaluation uses leave-one-center-out (LOCO) on NCT2013.
Adaptation is *episodic*: model parameters reset after each biopsy core.

Usage:
    python scripts/tta/evaluate_tta.py --test-center UVA --method baseline \
        --checkpoint checkpoints/tta/single_UVA/best_model.pth

    python scripts/tta/evaluate_tta.py --test-center UVA --method tent \
        --checkpoint checkpoints/tta/single_UVA/best_model.pth \
        --tta-lr 1e-3 --tta-steps 5

    python scripts/tta/evaluate_tta.py --test-center UVA --method ttt \
        --checkpoint checkpoints/tta/ttt_UVA/best_model.pth \
        --tta-lr 1e-3 --tta-steps 5

    python scripts/tta/evaluate_tta.py --test-center UVA --method denem \
        --checkpoint-dir checkpoints/tta/ensemble_UVA \
        --ensemble-size 5 --tta-lr 1e-2 --tta-steps 5 --mi-lambda 10
"""

import argparse
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import v2 as T

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.datasets.nct2013.cohort_selection import get_core_ids, get_metadata_table, get_patient_splits_by_center
from medAI.datasets.patches_dataset_wrapper import PatchesDatasetWrapper
from medAI.metrics import calculate_binary_classification_metrics
from medAI.transforms.normalization import InstanceNormalizeImage
from projects.seg_ttt.needle_trace_dataset_ttt import NeedleTraceImageFramesDataset

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from projects.seg_ttt.TTA_baselines.denem_models import TTTModel, create_resnet10, mutual_information_loss

try:
    import wandb
except ImportError:
    wandb = None


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────


def get_patch_transform():
    """Test-time patch transform (no augmentation)."""
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((256, 256)),
        InstanceNormalizeImage(),
        lambda p: torch.stack(p),
    ])


def transform_item(item, patch_t):
    return {
        "patches": patch_t(item["patches"]),
        "label": int(item["grade"] != "Benign"),
        "core_id": item["core_id"],
    }


def build_test_dataset(root_dir):
    """Build test dataset: all cores from the left-out center."""
    # _, _, test_patients = get_patient_splits_by_center(leave_out=test_center, val_size=0.2)
    # test_cores = get_core_ids(test_patients)

    test_ds = NeedleTraceImageFramesDataset(
            root_dir=root_dir,
            case_ids = case_ids,
            needle_mask_fname=(
                "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
            ),
        )

    # test_ds = BModeDatasetV1(core_ids=test_cores)
    test_ds = PatchesDatasetWrapper(
        test_ds,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="bmode",
        mask_keys=["needle_mask", "prostate_mask"],
        mask_thresholds=[0.6, 0.9],
        yield_one_patch_per_item=False,
        include_images=False,
        transform=partial(transform_item, patch_t=get_patch_transform()),
    )
    return test_ds


# ──────────────────────────────────────────────────────────────────────────────
# TENT
# ──────────────────────────────────────────────────────────────────────────────


def tent_adapt(model, patches, lr, steps):
    """
    Adapt by minimizing entropy of predictions.
    Only updates GroupNorm affine parameters (weight, bias).
    """
    params = []
    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            for p in module.parameters():
                p.requires_grad_(True)
                params.append(p)
        else:
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)

    optimizer = torch.optim.SGD(params, lr=lr)
    model.train()

    for _ in range(steps):
        logits = model(patches)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return F.softmax(model(patches), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# TTT
# ──────────────────────────────────────────────────────────────────────────────


def ttt_adapt(model, patches, lr, steps):
    """
    Adapt backbone using BYOL self-supervised loss on test patches.
    Classifier head is frozen; only backbone + BYOL heads are updated.
    """
    aug = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    adapt_params = (
        list(model.backbone.parameters())
        + list(model.projector.parameters())
        + list(model.predictor.parameters())
    )
    for p in adapt_params:
        p.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(adapt_params, lr=lr)
    model.train()

    for _ in range(steps):
        loss = model.forward_byol(aug(patches), aug(patches))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return F.softmax(model(patches), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# DEnEM
# ──────────────────────────────────────────────────────────────────────────────


def denem_adapt(models, patches, lr, steps, mi_lambda=10.0):
    """
    Adapt ensemble by minimizing:
        L = H(p_bar) + lambda * L_MI
    where p_bar is the mean softmax across members and L_MI encourages diversity.
    """
    all_params = []
    for m in models:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)
            all_params.append(p)

    optimizer = torch.optim.SGD(all_params, lr=lr)
    M = len(models)

    for _ in range(steps):
        all_logits = [m(patches) for m in models]
        all_probs = [F.softmax(l, dim=-1) for l in all_logits]

        # Marginal entropy
        p_bar = torch.stack(all_probs).mean(dim=0)
        h_bar = -(p_bar * torch.log(p_bar + 1e-8)).sum(dim=-1).mean()

        # MI diversity
        mi = torch.tensor(0.0, device=patches.device)
        n_pairs = 0
        for i in range(M):
            for j in range(i + 1, M):
                mi = mi + mutual_information_loss(all_logits[i], all_logits[j])
                n_pairs += 1
        if n_pairs > 0:
            mi = mi / n_pairs

        loss = h_bar + mi_lambda * mi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for m in models:
        m.eval()
    with torch.no_grad():
        return torch.stack([F.softmax(m(patches), dim=-1) for m in models]).mean(dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────


def load_single_model(path):
    model = create_resnet10()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.cuda().eval()


def load_ttt_model(path):
    model = TTTModel()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.cuda().eval()


def load_ensemble(checkpoint_dir, size):
    models = []
    for i in range(size):
        m = create_resnet10()
        m.load_state_dict(torch.load(Path(checkpoint_dir) / f"best_model_{i}.pth", map_location="cpu", weights_only=True))
        models.append(m.cuda().eval())
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(args):
    test_ds = build_test_dataset(args.test_center)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.method == "denem":
        source_models = load_ensemble(args.checkpoint_dir, args.ensemble_size)
    elif args.method == "ttt":
        source_model = load_ttt_model(args.checkpoint)
    else:
        source_model = load_single_model(args.checkpoint)

    all_preds, all_labels, all_core_ids = [], [], []

    for batch in tqdm(test_loader, desc=f"Evaluating ({args.method})"):
        B, N, C, H, W = batch["patches"].shape
        assert B == 1, "Episodic TTA requires batch_size=1"
        patches = batch["patches"].reshape(N, C, H, W).cuda()
        label = batch["label"].item()
        core_id = batch["core_id"][0] if isinstance(batch["core_id"], list) else batch["core_id"]

        if args.method == "baseline":
            with torch.no_grad():
                probs = F.softmax(source_model(patches), dim=-1)

        elif args.method == "tent":
            adapted = copy.deepcopy(source_model)
            probs = tent_adapt(adapted, patches, lr=args.tta_lr, steps=args.tta_steps)

        elif args.method == "ttt":
            adapted = copy.deepcopy(source_model)
            probs = ttt_adapt(adapted, patches, lr=args.tta_lr, steps=args.tta_steps)

        elif args.method == "denem":
            adapted = [copy.deepcopy(m) for m in source_models]
            probs = denem_adapt(adapted, patches, lr=args.tta_lr, steps=args.tta_steps, mi_lambda=args.mi_lambda)

        core_prob = probs.mean(dim=0).cpu()
        all_preds.append(core_prob)
        all_labels.append(label)
        all_core_ids.append(core_id)

    preds = torch.stack(all_preds).numpy()
    labels = np.array(all_labels)

    # All cores
    metrics_all = calculate_binary_classification_metrics(preds[:, 1], labels)

    # Filtered: exclude cancer cores with involvement < 40%
    meta = get_metadata_table()
    keep = np.ones(len(all_core_ids), dtype=bool)
    for i, cid in enumerate(all_core_ids):
        row = meta[meta.core_id == cid]
        if len(row) > 0:
            grade = row.iloc[0]["grade"]
            pct = row.iloc[0].get("pct_cancer", 0)
            if grade != "Benign" and pct < 40:
                keep[i] = False

    metrics_filt = (
        calculate_binary_classification_metrics(preds[keep, 1], labels[keep])
        if keep.sum() < len(keep)
        else metrics_all
    )

    REPORT_KEYS = [
        ("auc", "AUROC"),
        ("balanced_acc", "Balanced Acc"),
        ("sens", "Sensitivity"),
        ("spec", "Specificity"),
        ("sens_at_20_spe", "Sens@20Spe"),
        ("sens_at_40_spe", "Sens@40Spe"),
        ("sens_at_60_spe", "Sens@60Spe"),
        ("sens_at_80_spe", "Sens@80Spe"),
        ("f1", "F1"),
    ]

    print(f"\n{'='*60}")
    print(f"Method: {args.method} | Test center: {args.test_center}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'All':>10} {'Inv>=40%':>10}")
    print(f"{'-'*40}")
    for key, label in REPORT_KEYS:
        v_all = metrics_all.get(key, float("nan"))
        v_filt = metrics_filt.get(key, float("nan"))
        print(f"{label:<20} {v_all:>10.4f} {v_filt:>10.4f}")
    print(f"{'='*60}\n")

    # Save
    def _scalar(v):
        return float(v) if isinstance(v, (int, float, np.floating)) else None

    results = {
        "method": args.method,
        "test_center": args.test_center,
        "tta_lr": args.tta_lr,
        "tta_steps": args.tta_steps,
        "metrics_all": {k: _scalar(v) for k, v in metrics_all.items() if _scalar(v) is not None},
        "metrics_filtered": {k: _scalar(v) for k, v in metrics_filt.items() if _scalar(v) is not None},
    }
    save_dir = Path(args.save_dir) / args.method
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"results_{args.test_center}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    if wandb and args.log_wandb:
        log_dict = {
            "test_center": args.test_center,
            **{f"test_all/{k}": v for k, v in results["metrics_all"].items()},
            **{f"test_filtered/{k}": v for k, v in results["metrics_filtered"].items()},
        }
        wandb.log(log_dict)
        # Also set as summary so it shows in the runs table
        for k, v in log_dict.items():
            wandb.run.summary[k] = v

    return results


def main():
    p = argparse.ArgumentParser(description="TTA evaluation on NCT2013 (LOCO)")
    p.add_argument("--root-dir", type=str, required=True, choices=["UVA", "CRCEO", "PCC", "PMCC", "JH"])
    p.add_argument("--test-center", type=str, required=True, choices=["UVA", "CRCEO", "PCC", "PMCC", "JH"])
    p.add_argument("--method", type=str, required=True, choices=["baseline", "tent", "ttt", "denem"])
    p.add_argument("--checkpoint", type=str, default=None, help="Single model checkpoint (baseline/tent/ttt)")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Ensemble checkpoint dir (denem)")
    p.add_argument("--tta-lr", type=float, default=1e-2)
    p.add_argument("--tta-steps", type=int, default=5)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--mi-lambda", type=float, default=10.0)
    p.add_argument("--save-dir", type=str, default="results/tta")
    p.add_argument("--log-wandb", action="store_true")
    args = p.parse_args()

    if args.method in ("baseline", "tent", "ttt") and args.checkpoint is None:
        p.error(f"--checkpoint required for method={args.method}")
    if args.method == "denem" and args.checkpoint_dir is None:
        p.error("--checkpoint-dir required for method=denem")

    if wandb and args.log_wandb:
        wandb.init(project="PC-TTA", name=f"{args.method}_{args.test_center}", config=vars(args))

    evaluate(args)

    if wandb and args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
