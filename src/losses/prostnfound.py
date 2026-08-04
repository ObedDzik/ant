import argparse
from dataclasses import dataclass, field
import json
from typing import Callable
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange


class CancerDetectionValidRegionLoss(nn.Module):
    def __init__(
        self,
        base_loss: Callable = F.binary_cross_entropy_with_logits,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        label = data["label"].to(cancer_logits.device)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        return self.base_loss(predictions, labels)


class ProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight

    def forward(self, bag_of_logits, true_prop):
        probs = bag_of_logits.sigmoid()
        pred_prob = probs.mean()

        loss = -true_prop * pred_prob.log() - (1 - true_prop) * (1 - pred_prob).log()

        if true_prop > 0:
            loss = loss * self.pos_weight
        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
        if self.entropy_penalty_lambda:
            entropy = -probs * probs.log() - (1 - probs) * (1 - probs).log()
            loss = loss + self.entropy_penalty_lambda * entropy.mean()

        return loss

class ImprovedProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
        pos_weight: float = 4.0,  # Higher default
        focal_gamma: float = 2.0,  # Add focal focusing
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(self, bag_of_logits, true_prop):
        probs = bag_of_logits.sigmoid()
        pred_prob = probs.mean()
        
        # Clamp for numerical stability
        pred_prob = torch.clamp(pred_prob, min=1e-7, max=1-1e-7)
        
        # Separate positive and negative terms
        pos_loss = -true_prop * torch.log(pred_prob)
        neg_loss = -(1 - true_prop) * torch.log(1 - pred_prob)
        
        # Optional: Focal weighting (focuses on hard examples)
        if self.focal_gamma > 0:
            pt = true_prop * pred_prob + (1 - true_prop) * (1 - pred_prob)
            focal_weight = (1 - pt) ** self.focal_gamma
            pos_loss = pos_loss * focal_weight
            neg_loss = neg_loss * focal_weight
        
        # Apply pos_weight ONLY to positive term
        loss = self.pos_weight * pos_loss + neg_loss
        
        # Regularization
        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
        if self.entropy_penalty_lambda:
            entropy = -probs * torch.log(probs + 1e-7) - (1 - probs) * torch.log(1 - probs + 1e-7)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()
        
        return loss


class CancerDetectionMILLoss(nn.Module):
    def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):

        super().__init__()
        self.base_loss = base_loss
        self.treat_gg1_as_benign = treat_gg1_as_benign

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        batch_size = len(cancer_logits)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device)
        grade_group = data["grade_group"].to(cancer_logits.device)

        if self.treat_gg1_as_benign:
            involvement[grade_group == 1] = 0.0

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.tensor(0, device=cancer_logits.device)
        for i in range(batch_size):
            bag_i = predictions[batch_idx == i]
            involvement_i = involvement[i]

            loss = loss + self.base_loss(bag_i, involvement_i)

        return loss


class InvolvementL1Loss(nn.Module):
    def __init__(self, prostate_penalty=True, pos_weight=1):
        super().__init__()
        self.prostate_penalty = prostate_penalty
        self.pos_weight = pos_weight

    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        avg_prostate_heatmap_value = data["average_prostate_heatmap_value"]
        involvement = data["involvement"].to(device)
        cores_positive_for_patient = data["cores_positive_for_patient"]

        loss = torch.tensor(0, device=device)
        loss = loss + torch.nn.functional.l1_loss(
            avg_needle_heatmap_value, involvement, reduction="none"
        )
        for idx in range(B):
            if involvement[idx] > 0:
                loss[idx] *= self.pos_weight
        loss = loss.mean()

        if self.prostate_penalty:
            for idx in range(B):
                if cores_positive_for_patient[idx] == 0:
                    loss += avg_prostate_heatmap_value[idx]

        return loss


class InvolvementMSELoss(nn.Module):
    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        involvement = data["involvement"].to(device)

        loss = torch.nn.functional.mse_loss(avg_needle_heatmap_value, involvement)
        return loss


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


class ImageLevelClassificationLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, data):
        """
        Computes the image-level classification loss.
        """

        if "image_level_classification_outputs" not in data:
            return torch.tensor(0.0, device=data["label"].device)

        logits = data["image_level_classification_outputs"][0]

        if self.mode == "pca":
            labels = data["label"].to(logits.device)
        else:
            labels = (data["grade_group"] > 1).long().to(logits.device)

        # Compute the binary cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class SumLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, data):
        loss = self.losses[0](data) * self.weights[0]
        for i in range(1, len(self.losses)):
            loss += self.losses[i](data) * self.weights[i]
        return loss


class OutsideProstatePenaltyLoss(nn.Module):

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] < 0.5
            masks.append(mask)

        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.nn.L1Loss()(predictions, torch.zeros_like(predictions))

        return loss


def build_heatmap_loss(args):
    if args.loss == "needle_region_ce":
        return CancerDetectionValidRegionLoss()
    elif args.loss == "inv_l1":
        return InvolvementL1Loss(**args.loss_kw)
    elif args.loss == "inv_l1_v2":
        return InvolvementL1Loss(prostate_penalty=False)
    elif args.loss == "inv_mse":
        return InvolvementMSELoss(**args.loss_kw)
    elif args.loss == "mil_prop_bce":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(pos_weight=args.pos_weight),
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "mil_prop_bce_l1_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(0.001), treat_gg1_as_benign=args.treat_gg1_as_benign
        )
    elif args.loss == "mil_prop_bce_entropy_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(
                entropy_penalty_lambda=0.01, pos_weight=args.pos_weight
            ),
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "mil_prop_bce_entropy_reg_imp":
        return CancerDetectionMILLoss(
            base_loss=ImprovedProportionBCE(
                entropy_penalty_lambda=0.01, pos_weight=args.pos_weight
            ),
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "none":
        return None
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def build_loss(args):

    losses = []

    hmap_loss = build_heatmap_loss(args)

    if hmap_loss is not None:
        losses.append(hmap_loss)

    if args.add_image_clf:
        print(f"Adding image-level classification loss: {args.add_image_clf}")
        losses.append(ImageLevelClassificationLoss(mode=args.image_clf_mode))

    if args.outside_prostate_penalty:
        print("Adding outside prostate penalty loss.")
        losses.append(OutsideProstatePenaltyLoss())

    return SumLoss(losses)


@dataclass
class LossArgs:
    loss: str = "mil_prop_bce_entropy_reg"
    loss_kw: dict = field(default_factory=dict)
    add_image_clf: bool = False
    image_clf_mode: str = "cspca"
    outside_prostate_penalty: bool = False
    treat_gg1_as_benign: bool = False
    pos_weight: float = 1.0


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss", default="needle_region_ce")
    parser.add_argument(
        "--outside_prostate_penalty",
        action="store_true",
        default=False,
        help="Whether to penalize the model for making predictions outside the prostate region.",
    )
    return parser