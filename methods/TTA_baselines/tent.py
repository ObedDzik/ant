"""Tent baseline for test-time adaptation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    softmax_entropy,
)


class Tent(nn.Module):
    """Adapt model online by entropy minimization."""

    def __init__(self, model, optimizer, steps: int = 1, episodic: bool = False):
        super().__init__()
        if steps <= 0:
            raise ValueError("Tent requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        model_output = None
        for _ in range(self.steps):
            model_output = forward_and_adapt(x, self.model, self.optimizer)
        return model_output

    def reset(self):
        if self.model_state is None:
            raise RuntimeError("Cannot reset Tent without saved model state.")
        # Keep optimizer dynamics for continual online updates.
        self.model.load_state_dict(self.model_state, strict=True)

    def forward_no_adapt(self, x):
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    model_output = model(x)
    logits = extract_logits(model_output)

    loss = softmax_entropy(logits).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model_output


def setup_tent(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[Tent, list[str], str]:
    """Configure a model and wrap it with Tent.

    Returns:
        (tent_model, adapted_param_names, inferred_architecture)
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
    return Tent(model, optimizer, steps=steps, episodic=episodic), selected_names, arch