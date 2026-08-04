"""
Source model training for TTA experiments (Gilany et al., 2024).

Trains ResNet10 models on NCT2013 using leave-one-center-out (LOCO) protocol.
Supports:
  - Single model training (for baseline, TENT)
  - Single model + BYOL head joint training (for TTT)
  - Ensemble training with mutual information loss (for DEnEM)

Usage:
    # Single model for baseline/TENT
    python scripts/tta/train_source.py --test-center UVA --mode single --epochs 30

    # TTT (joint CE + BYOL training)
    python scripts/tta/train_source.py --test-center UVA --mode ttt --epochs 30

    # Ensemble for DEnEM (M=5 members, lambda=10)
    python scripts/tta/train_source.py --test-center UVA --mode ensemble \
        --ensemble-size 5 --mi-lambda 10 --epochs 30
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import v2 as T

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.datasets.nct2013.cohort_selection import select_cohort
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

#bash /project/6106383/obed/medproj/medAI/projects/seg_ttt/debug.sh 
# --root-dir /datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles --val-center UA 
# --mode ensemble --ensemble-size 5 --mi-lambda 10 --epochs 3

def get_patch_transform(augmentations=False):
    augs = (
        T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ])
        if augmentations
        else T.Identity()
    )
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((256, 256)),
        augs,
        InstanceNormalizeImage(),
        lambda p: torch.stack(p),
    ])


def transform_item(item, patch_t):
    return {
        "patches": patch_t(item["patches"]),
        "label": int(item["grade"] != "Benign"),
        "core_id": item["core_id"],
    }


def build_datasets(root_dir, val_center): #involvement_threshold_pct=40):
    """Build train/val patch datasets using LOCO protocol via select_cohort."""
    case_ids = None
    train_cores, val_cores, _ = select_cohort(
        mode='train_only',
        test_center=None,
        exclude_benign_cores_from_positive_patients=True,
        # involvement_threshold_pct=involvement_threshold_pct,
    )

    train_ds = BModeDatasetV1(train_cores)

    if val_center == 'UA' and root_dir == '/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe':
        print(f'Inside UA leg. Optimum center is {val_center}')
        raise ValueError(f'Root dir must be /datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles but got {root_dir}')

    if val_center == ('OL' or 'PU') and root_dir == '/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe':
        case_ids = [
            p
            for p in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, p)) and p[:2]==val_center
        ]
    if val_center == ('OL' or 'PU') and root_dir == '/datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles':
        print(f'Inside PU and OL leg. Optimum center is {val_center}')
        raise ValueError(f'Root dir must be /datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe but got {root_dir}')


    val_ds = NeedleTraceImageFramesDataset(
            root_dir=root_dir,
            case_ids = case_ids,
            needle_mask_fname=(
                "needle_mask.png" ),#if mode != "heatmap" else "needle_mask_full.png"
        )

    # val_ds = BModeDatasetV1(val_cores)

    if os.getenv("DEBUG"):
        train_ds = torch.utils.data.Subset(train_ds, list(range(20)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(10)))

    train_ds = PatchesDatasetWrapper(
        train_ds,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="bmode",
        mask_keys=["needle_mask", "prostate_mask"],
        mask_thresholds=[0.6, 0.9],
        yield_one_patch_per_item=True,
        include_images=False,
        transform=partial(transform_item, patch_t=get_patch_transform(augmentations=True)),
    )
    val_ds = PatchesDatasetWrapper(
        val_ds,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="image",
        mask_keys=["needle_mask", "prostate_mask"],
        mask_thresholds=[0.6, 0.9],
        yield_one_patch_per_item=False,
        include_images=False,
        transform=partial(transform_item, patch_t=get_patch_transform(augmentations=False)),
    )

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


def validate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            B, N, C, H, W = batch["patches"].shape
            patches = batch["patches"].reshape(B * N, C, H, W).cuda()
            logits = model(patches)
            probs = F.softmax(logits, dim=-1).reshape(B, N, -1).mean(dim=1)
            all_preds.append(probs.cpu())
            all_labels.append(batch["label"])
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return calculate_binary_classification_metrics(preds[:, 1], labels)


def validate_ensemble(models, val_loader):
    for m in models:
        m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            B, N, C, H, W = batch["patches"].shape
            patches = batch["patches"].reshape(B * N, C, H, W).cuda()
            probs = sum(F.softmax(m(patches), dim=-1) for m in models) / len(models)
            core_probs = probs.reshape(B, N, -1).mean(dim=1)
            all_preds.append(core_probs.cpu())
            all_labels.append(batch["label"])
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return calculate_binary_classification_metrics(preds[:, 1], labels)


# ──────────────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────────────

def train_ensemble(args):
    """Train a DEnEM ensemble: M ResNet10s with CE + MI diversity loss."""
    train_ds, val_ds = build_datasets(args.root_dir, args.val_center)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    M = args.ensemble_size
    models = [create_resnet10().cuda() for _ in range(M)]
    all_params = [p for m in models for p in m.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    save_dir = Path(args.save_dir) / f"ensemble_{args.test_center}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for m in models:
            m.train()
        total_loss = total_ce = total_mi = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            B, N, C, H, W = batch["patches"].shape
            patches = batch["patches"].reshape(B * N, C, H, W).cuda()
            labels = batch["label"].repeat_interleave(N).cuda()

            optimizer.zero_grad()
            all_logits = [m(patches) for m in models]
            ce_loss = sum(criterion(l, labels) for l in all_logits) / M

            mi_loss = torch.tensor(0.0, device="cuda")
            n_pairs = 0
            for i in range(M):
                for j in range(i + 1, M):
                    mi_loss = mi_loss + mutual_information_loss(all_logits[i], all_logits[j])
                    n_pairs += 1
            if n_pairs > 0:
                mi_loss = mi_loss / n_pairs

            loss = ce_loss + args.mi_lambda * mi_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_mi += mi_loss.item()

        n = len(train_loader)
        metrics = validate_ensemble(models, val_loader)
        print(
            f"Epoch {epoch+1}: loss={total_loss/n:.4f} "
            f"(CE={total_ce/n:.4f}, MI={total_mi/n:.4f}), "
            f"AUC={metrics['auc']:.4f}"
        )

        if wandb and args.log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": total_loss / n, "train/ce": total_ce / n, "train/mi": total_mi / n,
                **{f"val/{k}": v for k, v in metrics.items()},
            })

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            for i, m in enumerate(models):
                torch.save(m.state_dict(), save_dir / f"best_model_{i}.pth")

    for i, m in enumerate(models):
        torch.save(m.state_dict(), save_dir / f"final_model_{i}.pth")
    print(f"Done. Best val AUC: {best_auc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Source training for TTA experiments")
    p.add_argument("--root-dir", type=str, required=True,)
    p.add_argument("--val-center", type=str, required=True, choices=["UA", "OL"])
    p.add_argument("--mode", type=str, required=True, choices=["single", "ttt", "ensemble"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="/scratch/obed/denem")
    p.add_argument("--log-wandb", action="store_true")
    p.add_argument("--ttt-alpha", type=float, default=1.0, help="Weight for BYOL loss in TTT.")
    p.add_argument("--ensemble-size", type=int, default=5, help="Number of ensemble members (DEnEM).")
    p.add_argument("--mi-lambda", type=float, default=10.0, help="Weight for MI diversity loss (DEnEM).")
    args = p.parse_args()

    if wandb and args.log_wandb:
        wandb.init(project="PC-TTA", name=f"{args.mode}_source_{args.val_center}", config=vars(args))

    {"ensemble": train_ensemble}[args.mode](args)

    if wandb and args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
