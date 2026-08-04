import argparse
from collections import defaultdict
import json
import os
from argparse import ArgumentParser, Namespace
import time
import logging

import PIL
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import torch
from PIL import Image
import medAI
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.utils.accumulators import DataFrameCollector
from medAI.utils.argparse import UpdateDictAction
from medAI.modeling.prostnfound import ProstNFound
from medAI.factories.prostnfound.models import get_model
from medAI.modeling.setr import SETR

import numpy as np
import copy
import pandas as pd
from torch import nn
from tqdm import tqdm

from medAI.datasets.nct2013 import data_accessor
from medAI.modeling import list_models, create_model
from medAI.layers.masked_prediction_module import (
    MaskedPredictionModule,
)
from projects.seg_ttt.dataloader_ttt import get_dataloaders_from_args
from medAI.engine.prostnfound.evaluator import show_heatmap_prediction, show_heatmap_prediction_publication
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)

from projects.seg_ttt.infer_ttt import SupervisedTrainerWithSegTTT
from projects.seg_ttt.TransUnet.seg_model import FrozenSegmentationModel
from medAI.modeling.proside import Proside


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    else: 
        state = None

    train_args = Namespace(**state["args"])

    if args.model is None:
        args.model = train_args.model
    if args.model_kw is None:
        args.model_kw = train_args.model_kw

    if args.save_checkpoint:
        torch.save(state, os.path.join(args.output_dir, "checkpoint.pth"))
    # OmegaConf.save(
    #     state["args"],
    #     os.path.join(args.output_dir, "train_args.yaml"),
    # )
    if args.get("output_dir") is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        OmegaConf.save(
            args, os.path.join(args.output_dir, "test_config.yaml"), resolve=True
        )

    # OmegaConf.save(
    #     args,
    #     os.path.join(args.output_dir, "test_args.yaml"),
    # )
    model = get_model(OmegaConf.to_object(args))
    use_dino = args.get('use_dino', True)
    if use_dino:
        model = Proside(model, **args.get('model_kw', {}))
    model = ProstNFoundMeta(model, **args.get("metamodel", {}))

    # model = ProstNFoundMeta(create_model(args.model, **args.model_kw))
    print(model.load_state_dict(state["model"], strict=False))
    model.to(args.device)
    model.eval()
    if args.torch_compile:
        model = torch.compile(model)

    loaders = get_dataloaders_from_args(args.data)

    # maybe calibrate the temperature and bias of the model
    if args.calibration_mode == "pixel":
        do_calibration_pixel_wise_balanced_bce(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )
    elif args.calibration_mode == "bag":
        do_calibration_bag_wise(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )

    evaluator = Evaluator(
        log_images=False, include_patient_metrics=args.include_patient_metrics
    )
    accumulator = defaultdict(list)

    loader = loaders[args.split]

    ttt_seg_object=None
    if args.use_ttt:
        #TTT object
        print("\nTesting WITH TTT!!!!")
        frozen_segnet = FrozenSegmentationModel(
        checkpoint_path='/datasets/exactvu_pca/checkpoint_store/Paul/medAI/MicroSegNet.pth',
        img_size=224,
        device='cuda'
        )
        ttt_seg_object = SupervisedTrainerWithSegTTT(frozen_segnet, args)

        original_state = copy.deepcopy(model.state_dict())
        bn_state = save_bn_stats(model)
        dice_scores = []

    involvement_map = {}

    # warmup
    for _ in range(10):
        batch = next(iter(loader))
        model(batch)

    for i, data in enumerate(tqdm(loader)):
        # print(np.unique(data['prostate_mask']))

        # measure inference
        t0 = time.perf_counter()

        with torch.amp.autocast_mode.autocast(
            device_type=torch.device(args.device).type, enabled=args.use_amp
        ):

            if args.use_ttt:
                dice_score = ttt_seg_object.apply_segmentation_ttt(model, data, i)
                dice_scores.append(dice_score)

                model.eval()
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    with torch.inference_mode():
                        data = model(data)

                # Reset encoder after each batch unless persisting within val loop
                if not args.persist_encoder:
                    model.load_state_dict(original_state)
                    restore_bn_stats(model, bn_state)

            else:
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    with torch.inference_mode():
                        data = model(data)


            # with torch.inference_mode():
            #     data = model(data)

        if args.postprocess:
            cancer_logits = data.pop("cancer_logits")
            heatmap = cancer_logits[0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            
            import skimage
            # import cv2
# 
            # blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            # upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)

            blurred = skimage.filters.gaussian(heatmap, sigma=1.5)
            upsampled = skimage.transform.resize(blurred, (256, 256), order=1, anti_aliasing=True)
            upsampled = (upsampled * 255).astype(np.uint8)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]
        else:
            # get raw heatmap and also save as png
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap)

        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator["infer_time"].append(infer_time)

        if args.save_raw_heatmaps:
            # get raw heatmap and also save as png
            heatmap = Image.fromarray(heatmap)
            os.makedirs(os.path.join(args.output_dir, "raw_heatmaps"), exist_ok=True)
            heatmap.save(
                os.path.join(
                    args.output_dir, "raw_heatmaps", data["core_id"][0] + ".png"
                )
            )

        if args.save_rendered_heatmaps:

            patient_id = data['patient_id'][0]
            core_id = data['core_id'][0]
        
            output_file = os.path.join(
                args.output_dir, 
                "heatmaps", 
                patient_id,
                f"{core_id}.{args.save_format}"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # show_heatmap_prediction(data)
            show_heatmap_prediction_publication(data)
            plt.savefig(
                output_file,
                format=args.save_format,
            )
            plt.close()

        for core_id, inv in zip(data['core_id'], data['involvement']):
            core_id = core_id if isinstance(core_id, str) else core_id.item()
            inv = inv.item() if hasattr(inv, 'item') else float(inv)
            involvement_map[core_id] = inv

        evaluator(data)

    table = evaluator.accumulator.compute()
    table['true_involvement'] = table['core_id'].map(involvement_map)
    table.to_csv(os.path.join(args.output_dir, "metrics_by_core.csv"))

    metrics = evaluator.aggregate_metrics()
    metrics["infer_time"] = np.array(accumulator["infer_time"]).mean()
    metrics = {k: float(v) for k, v in metrics.items()}

    print(json.dumps(metrics, indent=4))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def save_bn_stats(model):
    bn_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_state[name] = {
                "running_mean": module.running_mean.clone() if module.running_mean is not None else None,
                "running_var": module.running_var.clone() if module.running_var is not None else None,
                "num_batches_tracked": module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
                "track_running_stats": module.track_running_stats,
            }
    return bn_state

def restore_bn_stats(model, bn_state):
    for name, module in model.named_modules():
        if name in bn_state:
            state = bn_state[name]

            if state["running_mean"] is not None:
                module.running_mean.data.copy_(state["running_mean"])
            if state["running_var"] is not None:
                module.running_var.data.copy_(state["running_var"])
            if state["num_batches_tracked"] is not None:
                module.num_batches_tracked.data.copy_(state["num_batches_tracked"])

            module.track_running_stats = state["track_running_stats"]


def do_calibration_pixel_wise_balanced_bce(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # extract all pixel predictions from val loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loaders["val"], device
    )
    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    from torch.optim import LBFGS

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(temp)

    optim = LBFGS(params, lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= 1.6

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(temp)
    model.bias.data.copy_(bias)


def do_calibration_bag_wise(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    bags_of_logits, involvement, label = extract_all_bag_predictions(
        model, loaders["val"], device
    )

    # fit temperature and bias to center and scale the predictions
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))

    from torch.optim import LBFGS

    pos_weight = (1 - label).sum() / label.sum()

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(log_temp)

    optim = LBFGS(params, lr=1e-1, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for bag_i, involvement_i, label_i in zip(bags_of_logits, involvement, label):
            bag_i = bag_i / log_temp.exp() + bias
            bag_i = bag_i.sigmoid()
            bag_i_mean = bag_i.mean()
            loss_i = (
                -involvement_i * bag_i_mean.log()
                - (1 - involvement_i) * (1 - bag_i_mean).log()
            )
            if label_i:
                loss_i = loss_i * pos_weight
            loss = loss + loss_i
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(log_temp.exp())
    model.bias.data.copy_(bias)


@torch.no_grad()
def extract_all_bag_predictions(model, loader, device):

    bags_of_logits = []
    involvement = []
    label = []

    for data in tqdm(loader, f"Running model..."):
        data = model(data)
        bags_of_logits.extend(
            get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
        )
        involvement.append(data["involvement"].to(device))
        label.append(data["label"].to(device))

    involvement = torch.cat(involvement)
    label = torch.cat(label)

    return bags_of_logits, involvement, label


@torch.no_grad()
def extract_heatmap_and_data(model, batch, device):
    bmode = batch.pop("bmode").to(device)
    needle_mask = batch.pop("needle_mask").to(device)
    prostate_mask = batch.pop("prostate_mask").to(device)

    psa = batch["psa"].to(device)
    age = batch["age"].to(device)
    label = batch["label"].to(device)
    family_history = batch["family_history"].to(device)
    anatomical_location = batch["loc"].to(device)

    core_id = batch["core_id"][0]

    B = len(bmode)
    task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

    heatmap_logits = model(
        bmode,
        task_id=task_id,
        anatomical_location=anatomical_location,
        psa=psa,
        age=age,
        family_history=family_history,
        prostate_mask=prostate_mask,
        needle_mask=needle_mask,
    ).cpu()

    heatmap_logits = heatmap_logits[0, 0].sigmoid().cpu().numpy()
    bmode = bmode[0, 0].cpu().numpy()
    prostate_mask = prostate_mask[0, 0].cpu().numpy()
    needle_mask = needle_mask[0, 0].cpu().numpy()
    core_id = core_id

    return heatmap_logits, bmode, prostate_mask, needle_mask, core_id


def extract_all_pixel_predictions(model, loader, device):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(device)

    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            data = model(data)

            prostate_mask = data["prostate_mask"].to(device)
            needle_mask = data["needle_mask"].to(device)
            heatmap_logits = data["cancer_logits"]
            label = data["label"]
            core_id = data["core_id"]

            # compute predictions
            masks = (prostate_mask > 0.5) & (needle_mask > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


def get_core_predictions_from_pixel_predictions(pixel_preds, pixel_labels, core_ids):
    data = []
    for core in np.unique(core_ids):
        mask = core_ids == core
        core_pred = pixel_preds[mask].sigmoid().mean().item()
        core_label = pixel_labels[mask][0].item()
        data.append({"core_id": core, "core_pred": core_pred, "core_label": core_label})

    df = pd.DataFrame(data)
    return df


class ProstNFoundMeta(nn.Module):
    """Wraps a model to perform forward pass with ProstNFound style training

    Args:
        model: The model to wrap.
        mask_output_key: The key to use for the mask output (if the model outputs a dictionary of tensors)
    """

    def __init__(self, model: nn.Module, mask_output_key=None):
        super().__init__()
        self.model = model
        self.mask_output_key = mask_output_key

        if isinstance(self.model, ProstNFound):
            logging.info(f"Model ProstNFound with prompts {self.model.prompts}")

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False):
        # extracting relevant data from the batch
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        # print(torch.unique(prostate_mask))
        if "rf" in data:
            rf = data["rf"].to(self.device)
        else:
            rf = None

        B = len(bmode)

        # Wrapped forward pass
        if isinstance(self.model, ProstNFound):
            prompts = {}
            for prompt_name in self.model.prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, rf, prostate_mask, needle_mask, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )

        elif isinstance(self.model, Proside):
            prompts = {}
            for prompt_name in self.model.discrete_prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )
            
        else:
            model_outputs = self.model(bmode)
            if isinstance(model_outputs, dict):
                cancer_logits = model_outputs[self.mask_output_key]
            else:
                cancer_logits = self.model(bmode)

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps:
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            import cv2

            blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]

        return data

    def get_params_groups(self):
        if isinstance(self.model, SETR):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "head" in name:
                    warmup_parameters.append(param)
                else:
                    encoder_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif isinstance(self.model, ProstNFound):
            return self.model.get_params_groups()

        elif hasattr(self.model, "image_encoder"):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "image_encoder" in name:
                    encoder_parameters.append(param)
                else:
                    warmup_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "get_params_groups"):
            return self.model.get_params_groups()

        else:
            from itertools import chain

            encoder_parameters = []
            warmup_parameters = self.model.parameters()
            cnn_parameters = []

            return encoder_parameters, warmup_parameters, cnn_parameters

def load_config(config_path, options):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(options))

    # # deal with issues from old schemas
    # if cfg.tracked_metric == "val/core_auc_high_involvement":
    #     warn("`val/core_auc_high_involvement` is deprecated - use `val/auc` instead")
    #     cfg.tracked_metric = "val/auc"

    return cfg

if __name__ == "__main__":
    p = ArgumentParser(description="Train ProstNFound model")
    p.add_argument(
        "--config", "-c", help="Path to config file (located in cfg/train/...)"
    )
    p.add_argument("options", nargs=argparse.REMAINDER)
    args = p.parse_args()
    cfg = load_config(args.config, args.options)

    main(cfg)
