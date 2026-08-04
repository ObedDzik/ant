"""CoTTA baseline for test-time adaptation.

Adapted from the official implementation:
    Wang et al., CVPR 2022 — Continual Test-Time Domain Adaptation
    https://github.com/qinenergy/cotta

Key design points from the official code:
  - Two separate teacher models: model_ema (EMA teacher) and model_anchor (frozen source).
  - model_anchor confidence gates whether augmentation-averaged predictions are used.
  - Loss is symmetric cross-entropy between student logits and EMA teacher predictions.
  - Stochastic restoration uses p=0.001 and only touches weight/bias parameters.
  - model_ema is set to train() before each forward to allow BN to use batch statistics.
  - The EMA teacher prediction (outputs_ema) is returned, not a separate clean forward.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    load_model_and_optimizer,
)


@torch.jit.script
def softmax_entropy_symmetric(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy between student (x) and EMA teacher (x_ema).

    Matches the official CoTTA loss exactly:
        -0.5 * (p_ema * log p_student) - 0.5 * (p_student * log p_ema)
    """
    return (
        -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1)
        - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    )


def update_ema_variables(
    ema_model: nn.Module, model: nn.Module, alpha_teacher: float
) -> nn.Module:
    """Update EMA teacher weights from student.

    Matches the official update:
        ema_param = alpha * ema_param + (1 - alpha) * param
    Note: the official code iterates over .parameters() (not state_dict),
    so buffers (e.g. BN running stats) are NOT updated in the teacher.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1.0 - alpha_teacher)
    return ema_model


# def augment_input(x: torch.Tensor) -> torch.Tensor:
#     """Stochastic augmentation matching the spirit of the official KATANA transforms.

#     The official code uses a full augmentation pipeline (ColorJitter, RandomAffine,
#     GaussianBlur, GaussianNoise, horizontal flip). We apply equivalent lightweight
#     augmentations suitable for arbitrary input tensors without PIL dependency.
#     """
#     x_aug = x.clone()
#     x_aug = x_aug + torch.randn_like(x_aug) * 0.005  # Gaussian noise (std matches official)
#     if torch.rand(1).item() > 0.5:
#         x_aug = torch.flip(x_aug, dims=[-1])          # random horizontal flip
#     return x_aug.clamp(0.0, 1.0)

def augment_input(x: dict) -> dict:
    """
    CoTTA augmentations following Wang et al. CVPR 2022.
    Operates only on the 'bmode' image key, preserving all other dict entries.
    """
    x_aug = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    bmode = x_aug['bmode']
    # Gaussian noise
    bmode = bmode + torch.randn_like(bmode) * 0.005
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        bmode = torch.flip(bmode, dims=[-1])
    # # Random vertical flip
    # if torch.rand(1).item() > 0.5:
    #     bmode = torch.flip(bmode, dims=[-2])
    x_aug['bmode'] = bmode
    return x_aug


class CoTTA(nn.Module):
    """Continual Test-Time Adaptation.

    Faithfully adapted from the official qinenergy/cotta implementation.
    Three mechanisms:
      1. Anchor-gated augmentation-averaged EMA predictions as pseudo-labels.
      2. Symmetric cross-entropy loss between student and EMA teacher.
      3. Stochastic restoration of weight/bias params to source values (p=0.001).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        alpha_teacher: float = 0.999,
        restoration_factor: float = 0.001,        # official default is 0.001
        augmentation_multiplicity: int = 32,
        aug_confidence_threshold: float = 0.1,    # official gate threshold
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("CoTTA requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.alpha_teacher = alpha_teacher
        self.restoration_factor = restoration_factor
        self.augmentation_multiplicity = augmentation_multiplicity
        self.aug_confidence_threshold = aug_confidence_threshold

        # Save source state for stochastic restore and episodic reset.
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

        # Two teacher models matching the official copy_model_and_optimizer return:
        #   model_ema   — EMA teacher, weights detached, updated online.
        #   model_anchor — frozen source copy, used only for confidence gating.
        self.model_ema = self._build_ema_model()
        self.model_anchor = self._build_anchor_model()

    def _build_ema_model(self) -> nn.Module:
        """EMA teacher — weights detached, updated via EMA each step."""
        ema = copy.deepcopy(self.model)
        for param in ema.parameters():
            param.detach_()
        return ema

    def _build_anchor_model(self) -> nn.Module:
        """Anchor model — completely frozen source copy for confidence gating."""
        anchor = copy.deepcopy(self.model)
        for param in anchor.parameters():
            param.requires_grad_(False)
        anchor.eval()
        return anchor

    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = self._forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def _forward_and_adapt(self, x: torch.Tensor):
        """One CoTTA adaptation step. Matches official forward_and_adapt closely."""

        # --- Student forward on clean input ---
        outputs = self.model(x)
        student_logits = extract_logits(outputs)

        # --- EMA teacher set to train() so BN uses batch stats (official behaviour) ---
        self.model_ema.train()

        # --- Anchor confidence: max softmax prob from frozen source model ---
        with torch.no_grad():
            anchor_logits = extract_logits(self.model_anchor(x))
            anchor_prob = F.softmax(anchor_logits, dim=1).max(1)[0]  # [B]

        # --- Standard (non-augmented) EMA teacher prediction ---
        with torch.no_grad():
            standard_ema_logits = extract_logits(self.model_ema(x))

        # --- Gate: use augmentation-averaged predictions only when confidence is low ---
        to_aug = anchor_prob.mean(0) < self.aug_confidence_threshold

        if to_aug:
            outputs_emas = []
            for _ in range(self.augmentation_multiplicity):
                aug_logits = extract_logits(self.model_ema(augment_input(x))).detach()
                outputs_emas.append(aug_logits)
            ema_logits = torch.stack(outputs_emas).mean(0)
        else:
            ema_logits = standard_ema_logits

        # --- Symmetric cross-entropy loss: student vs EMA teacher ---
        loss = softmax_entropy_symmetric(student_logits, ema_logits.detach()).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # --- EMA teacher update from the now-updated student ---
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema,
            model=self.model,
            alpha_teacher=self.alpha_teacher,
        )

        # --- Stochastic restore: randomly reset weight/bias to source values ---
        # Matches official: only weight/bias, only requires_grad params, p=0.001.
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias") and param.requires_grad:
                    full_name = (
                        f"{module_name}.{param_name}" if module_name else param_name
                    )
                    if full_name in self.model_state:
                        mask = (
                            torch.rand(param.shape, device=param.device)
                            < self.restoration_factor
                        ).float()
                        source_val = self.model_state[full_name].to(param.device)
                        with torch.no_grad():
                            param.data = source_val * mask + param.data * (1.0 - mask)

        # --- Return EMA teacher prediction (matches official: returns outputs_ema) ---
        return self._wrap_ema_output(ema_logits, outputs)

    def _wrap_ema_output(self, ema_logits: torch.Tensor, reference_output):
        """Return EMA logits in the same format as the model's original output."""
        if isinstance(reference_output, torch.Tensor):
            return ema_logits
        if isinstance(reference_output, dict):
            # Reconstruct dict with EMA logits substituted in at the logits key
            out = {k: v for k, v in reference_output.items()}
            # Find which key holds the logits by checking what extract_logits reads
            logits_key = "image_level_classification_outputs"
            if logits_key in out:
                original = out[logits_key]
                if isinstance(original, (list, tuple)):
                    inner = list(original)
                    inner[0] = ema_logits
                    out[logits_key] = type(original)(inner)
                else:
                    out[logits_key] = ema_logits
            return out
        if isinstance(reference_output, (list, tuple)):
            out = list(reference_output)
            out[0] = ema_logits
            return type(reference_output)(out)
        return ema_logits

    def reset(self):
        """Reset student, EMA teacher, and anchor model to source state."""
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset CoTTA without saved states.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        # Rebuild both teacher models from the freshly restored student.
        self.model_ema = self._build_ema_model()
        self.model_anchor = self._build_anchor_model()

    def forward_no_adapt(self, x: torch.Tensor):
        return self.model(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_cotta(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    alpha_teacher: float = 0.999,
    restoration_factor: float = 0.001,
    augmentation_multiplicity: int = 32,
    aug_confidence_threshold: float = 0.1,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[CoTTA, list[str], str]:
    """Configure a model and wrap it with CoTTA.

    Note: The official CoTTA updates ALL model parameters (weight + bias across
    all modules). configure_model_for_adaptation handles BN running-stats nulling;
    collect_adaptation_params selects which params the optimizer updates.

    Returns:
        (cotta_model, adapted_param_names, inferred_architecture)
    """
    optimizer_kwargs = optimizer_kwargs or {}

    model, selected_names, arch = configure_model_for_adaptation(
        model=model,
        architecture=architecture,
        layer_selection=layer_selection,
    )
    check_adaptation_ready(model)

    params, _, _ = collect_adaptation_params(
        model=model,
        architecture=arch,
        layer_selection=layer_selection,
    )
    optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
    cotta = CoTTA(
        model=model,
        optimizer=optimizer,
        steps=steps,
        episodic=episodic,
        alpha_teacher=alpha_teacher,
        restoration_factor=restoration_factor,
        augmentation_multiplicity=augmentation_multiplicity,
        aug_confidence_threshold=aug_confidence_threshold,
    )
    return cotta, selected_names, arch