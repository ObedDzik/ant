"""Common utilities for test-time adaptation baselines.

This module keeps baseline implementations model-agnostic:
- Supports models returning logits tensor directly.
- Supports models returning tuples/lists where index 0 is logits.
- Selects adaptation layers based on architecture (CNN vs transformer).
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


NORM_CNN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
NORM_TRANSFORMER_TYPES = (nn.LayerNorm,)


def extract_logits(model_output):
    cls = model_output["image_level_classification_outputs"]
    if isinstance(cls, (list, tuple)):
        cls = cls[0]  # [B, 2]
    return cls  

@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return per-sample predictive entropy from logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)  # [B]

def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    """Deep-copy model and optimizer states for reset."""
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state,
    optimizer_state,
):
    """Restore model and optimizer states."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def infer_architecture(model: nn.Module) -> str:
    """Infer architecture family from available normalization layers."""
    has_bn_like = any(isinstance(m, NORM_CNN_TYPES) for m in model.modules())
    has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())

    if has_ln and not has_bn_like:
        return "transformer"
    if has_bn_like and not has_ln:
        return "cnn"
    if has_ln and has_bn_like:
        return "hybrid"
    return "generic"


def _module_block_index(module_name: str) -> Optional[int]:
    patterns = (
        r"(?:blocks?|stages?|layers?)\.(\d+)",
        r"encoder\.layer\.(\d+)",
        r"layer(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    return None


def _max_block_index(module_names: Iterable[str]) -> Optional[int]:
    max_idx = None
    for name in module_names:
        idx = _module_block_index(name)
        if idx is not None:
            max_idx = idx if max_idx is None else max(max_idx, idx)
    return max_idx


def _is_late_block(module_name: str, max_idx: Optional[int]) -> bool:
    if max_idx is None:
        return True
    idx = _module_block_index(module_name)
    if idx is None:
        return True
    cutoff = int(math.floor((2.0 * max_idx) / 3.0))
    return idx >= cutoff


def _is_classifier_or_head_name(module_name: str) -> bool:
    if not module_name:
        return False
    keywords = ("head", "classifier", "fc", "logits", "proj")
    exact_names = {"norm", "fc_norm", "ln_post", "norm_head"}
    return module_name in exact_names or any(k in module_name for k in keywords)


def _norm_types_for_architecture(arch: str):
    if arch == "cnn":
        return NORM_CNN_TYPES
    if arch == "transformer":
        return NORM_TRANSFORMER_TYPES
    if arch == "hybrid":
        return NORM_CNN_TYPES + NORM_TRANSFORMER_TYPES
    return NORM_CNN_TYPES + NORM_TRANSFORMER_TYPES


def collect_adaptation_params(
    model: nn.Module,
    architecture: str = "auto",
    layer_selection: str = "auto",
) -> Tuple[List[nn.Parameter], List[str], str]:
    """Collect trainable parameters for adaptation.

    Args:
        model: Classification model.
        architecture: One of {"auto", "cnn", "transformer", "hybrid"}.
        layer_selection: One of {"auto", "all", "late"}.
            - "all": all eligible normalization layers.
            - "late": only later blocks/stages for faster, more stable adaptation.
            - "auto": uses "late" for transformers, "all" otherwise.
    """
    arch = infer_architecture(model) if architecture == "auto" else architecture
    selection = layer_selection
    if selection == "auto":
        selection = "all" if arch == "transformer" else "all"

    params: List[nn.Parameter] = []
    names: List[str] = []
    module_names = [name for name, _ in model.named_modules()]
    max_idx = _max_block_index(module_names)
    allowed_norms = _norm_types_for_architecture(arch)

    for module_name, module in model.named_modules():
        if not isinstance(module, allowed_norms):
            continue
        if arch == "transformer" and _is_classifier_or_head_name(module_name):
            continue
        if selection == "late" and not _is_late_block(module_name, max_idx):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            if param_name in {"weight", "bias"}:
                params.append(param)
                names.append(f"{module_name}.{param_name}" if module_name else param_name)

    return params, names, arch


def configure_model_for_adaptation(
    model: nn.Module,
    architecture: str = "auto",
    layer_selection: str = "auto",
) -> Tuple[nn.Module, List[str], str]:
    """Configure model state for test-time adaptation."""
    model.train()
    model.requires_grad_(False)

    _, selected_names, arch = collect_adaptation_params(
        model=model,
        architecture=architecture,
        layer_selection=layer_selection,
    )
    selected_set = set(selected_names)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in selected_set:
                param.requires_grad_(True)

        # Force batch-statistics usage for BN during adaptation.
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if any(
                (f"{module_name}.{p}" if module_name else p) in selected_set
                for p in ("weight", "bias")
            ):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    return model, selected_names, arch


def check_adaptation_ready(model: nn.Module):
    """Validate model mode and trainable-parameter setup."""
    if not model.training:
        raise AssertionError("Adaptation requires train mode. Call model.train().")

    grad_flags = [p.requires_grad for p in model.parameters()]
    if not any(grad_flags):
        raise AssertionError("No trainable parameters selected for adaptation.")
    if all(grad_flags):
        raise AssertionError("All model params are trainable; adaptation should be selective.")