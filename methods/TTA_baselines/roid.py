"""ROID baseline for robust online adaptation.

Based on the original ROID implementation from test-time-adaptation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    infer_architecture,
    load_model_and_optimizer,
    softmax_entropy,
)


# def _replace_logits(model_output, new_logits: torch.Tensor):
#     if isinstance(model_output, torch.Tensor):
#         return new_logits
#     if isinstance(model_output, tuple) and len(model_output) > 0:
#         return (new_logits,) + tuple(model_output[1:])
#     if isinstance(model_output, list) and len(model_output) > 0:
#         return [new_logits] + list(model_output[1:])
#     return new_logits

def _replace_logits(model_output, logits):
    """Replace image_level_classification_outputs in the output dict with corrected logits."""
    if isinstance(model_output, dict):
        out = {k: v for k, v in model_output.items()}  # shallow copy
        # image_level_classification_outputs is a list of tensors
        if isinstance(out.get('image_level_classification_outputs'), list):
            out['image_level_classification_outputs'] = [logits]
        else:
            out['image_level_classification_outputs'] = logits
        return out
    return logits  # fallback for plain tensor models


def _soft_likelihood_ratio(logits: torch.Tensor, clip: float = 0.99, eps: float = 1e-5) -> torch.Tensor:
    probs = logits.softmax(1)
    probs = torch.clamp(probs, min=0.0, max=clip)
    return -(probs * torch.log((probs / (torch.ones_like(probs) - probs)) + eps)).sum(1)


def _symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return -(1.0 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (
        x.softmax(1) * x_ema.log_softmax(1)
    ).sum(1)


def _safe_minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    v_min = values.min()
    v_max = values.max()
    denom = (v_max - v_min).clamp_min(1e-12)
    return (values - v_min) / denom


def _build_tta_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Lambda(lambda t: torch.clip(t, 0.0, 1.0)),
            transforms.ColorJitter(
                brightness=(0.6, 1.4),
                contrast=(0.7, 1.3),
                saturation=(0.5, 1.5),
                hue=(-0.06, 0.06),
            ),
            transforms.Pad(padding=int(image_size / 2), padding_mode="reflect"),
            transforms.RandomAffine(
                degrees=15,
                translate=(1.0 / 16.0, 1.0 / 16.0),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(size=image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda t: torch.clip(t, 0.0, 1.0)),
        ]
    )


class ROID(nn.Module):
    """ROID adaptation with weighted SLR loss and optional consistency."""

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        use_weighting: bool = True,
        use_prior_correction: bool = True,
        use_consistency: bool = True,
        momentum_src: float = 0.99,
        momentum_probs: float = 0.9,
        temperature: float = 1.0 / 3.0,
        image_size: int = 224,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("ROID requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.use_weighting = use_weighting
        self.use_prior_correction = use_prior_correction
        self.use_consistency = use_consistency
        self.momentum_src = momentum_src
        self.momentum_probs = momentum_probs
        self.temperature = temperature
        self.tta_transform = _build_tta_transform(image_size=image_size)

        with torch.no_grad():
            dummy_num_classes = None
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    dummy_num_classes = module.out_features
            self.num_classes = dummy_num_classes

        self.class_probs_ema = None
        self.src_model = deepcopy(self.model)
        for p in self.src_model.parameters():
            p.detach_()

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def _update_class_probs(self, probs_mean: torch.Tensor):
        if self.class_probs_ema is None:
            self.class_probs_ema = probs_mean.detach().clone()
        else:
            self.class_probs_ema = self.momentum_probs * self.class_probs_ema + (
                1.0 - self.momentum_probs
            ) * probs_mean.detach()

    def _slice_input(self, x, mask):
        """Slice input by boolean mask, works for both tensor and dict."""
        if isinstance(x, dict):
            cpu_mask = mask.cpu()
            result = {}
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v[mask.to(v.device)]
                elif isinstance(v, list):
                    if len(v) == len(cpu_mask):
                        # one element per sample — slice by mask
                        result[k] = [v[i] for i, m in enumerate(cpu_mask) if m]
                    else:
                        # shared/scalar list (e.g. length 1) — pass through as-is
                        result[k] = v
                else:
                    result[k] = v
            return result
        return x[mask]

    def _tta_transform_dict(self, x: dict) -> dict:
        """Apply tta_transform only to 'bmode', pass everything else through."""
        out = {k: v for k, v in x.items()}
        bmode = x['bmode']  # (B, C, H, W)
        augmented = []
        for img in bmode:
            augmented.append(self.tta_transform(img))  # operates per-image tensor
        out['bmode'] = torch.stack(augmented, dim=0)
        return out

    def _merge_with_source(self):
        with torch.no_grad():
            for param, src_param in zip(self.model.parameters(), self.src_model.parameters()):
                if param.requires_grad:
                    param.data = self.momentum_src * param.data + (1.0 - self.momentum_src) * src_param.data.to(param.device)

    def _loss_calculation(self, x):
        model_output = self.model(x)
        logits = extract_logits(model_output)
        probs = logits.softmax(1)
        batch_size = max(1, logits.shape[0])

        if self.class_probs_ema is None:
            self.class_probs_ema = probs.mean(0).detach().clone()

        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        weights = torch.ones(logits.shape[0], device=logits.device)

        if self.use_weighting:
            with torch.no_grad():
                weights_div = 1.0 - F.cosine_similarity(
                    self.class_probs_ema.unsqueeze(0), probs, dim=1
                )
                weights_div = _safe_minmax_normalize(weights_div)
                mask = weights_div < weights_div.mean()

                weights_cert = -softmax_entropy(logits)
                weights_cert = _safe_minmax_normalize(weights_cert)
                weights = torch.exp(weights_div * weights_cert / max(self.temperature, 1e-12))
                weights[mask] = 0.0

                self._update_class_probs(probs.mean(0))

        loss_out = _soft_likelihood_ratio(logits)
        if self.use_weighting:
            selected = ~mask
            if selected.any():
                loss = (loss_out[selected] * weights[selected]).sum() / batch_size
            else:
                loss = logits.new_tensor(0.0)
        else:
            selected = torch.ones_like(mask, dtype=torch.bool)
            loss = loss_out.sum() / batch_size

        if self.use_consistency and selected.any():
            logits_ref = logits[selected].detach()
            x_selected = self._slice_input(x, selected)
            x_aug = self._tta_transform_dict(x_selected) if isinstance(x, dict) else self.tta_transform(x_selected)
            aug_output = self.model(x_aug)
            aug_logits = extract_logits(aug_output)
            sce = _symmetric_cross_entropy(aug_logits, logits_ref)
            if self.use_weighting:
                loss = loss + (sce * weights[selected]).sum() / batch_size
            else:
                loss = loss + sce.sum() / batch_size

        return model_output, logits, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        model_output, logits, loss = self._loss_calculation(x)

        if not torch.isnan(loss):
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        self._merge_with_source()

        if self.use_prior_correction:
            with torch.no_grad():
                prior = logits.softmax(1).mean(0)
                smooth = max(1.0 / logits.shape[0], 1.0 / logits.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1.0 + smooth * logits.shape[1])
                logits = logits * smoothed_prior
                return _replace_logits(model_output, logits)

        return model_output

    def forward(self, x):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x)
        return out

    def forward_no_adapt(self, x):
        return self.model(x)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset ROID without saved states.")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.class_probs_ema = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_roid(
    model: nn.Module,
    lr: float = 2.5e-4,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    use_weighting: bool = True,
    use_prior_correction: bool = True,
    use_consistency: bool = True,
    momentum_src: float = 0.99,
    momentum_probs: float = 0.9,
    temperature: float = 1.0 / 3.0,
    image_size: int = 224,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs: dict | None = None,
) -> Tuple[ROID, list[str], str]:
    """Configure a model and wrap it with ROID."""
    optimizer_kwargs = optimizer_kwargs or {"momentum": 0.9}

    model, selected_names, arch = configure_model_for_adaptation(
        model=model,
        architecture=architecture,
        layer_selection=layer_selection,
    )
    check_adaptation_ready(model)
    model.eval()

    params, _, _ = collect_adaptation_params(
        model=model,
        architecture=arch,
        layer_selection=layer_selection,
    )
    optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
    roid = ROID(
        model=model,
        optimizer=optimizer,
        steps=steps,
        episodic=episodic,
        use_weighting=use_weighting,
        use_prior_correction=use_prior_correction,
        use_consistency=use_consistency,
        momentum_src=momentum_src,
        momentum_probs=momentum_probs,
        temperature=temperature,
        image_size=image_size,
    )
    return roid, selected_names, arch