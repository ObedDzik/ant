"""SAR baseline for test-time adaptation."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    load_model_and_optimizer,
    softmax_entropy,
)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho={rho}; should be non-negative.")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale_eps = 1e-12
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + scale_eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure for step().")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def _update_ema(ema, value):
    if ema is None:
        return value
    return 0.9 * ema + 0.1 * value


class SAR(nn.Module):
    """Sharpness-Aware and Reliable adaptation."""

    def __init__(
        self,
        model,
        optimizer: SAM,
        steps: int = 1,
        episodic: bool = False,
        margin_e0: float = 0.4 * math.log(1000),
        reset_constant_em: float = 0.2,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("SAR requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.margin_e0 = margin_e0
        self.reset_constant_em = reset_constant_em
        self.ema = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        model_output = None
        for _ in range(self.steps):
            model_output, ema, need_reset = forward_and_adapt_sar(
                x=x,
                model=self.model,
                optimizer=self.optimizer,
                margin=self.margin_e0,
                reset_constant=self.reset_constant_em,
                ema=self.ema,
            )
            self.ema = ema
            if need_reset:
                self.reset()
        return model_output

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset SAR without saved states.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.ema = None

    def forward_no_adapt(self, x):
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.enable_grad()
def forward_and_adapt_sar(x, model, optimizer: SAM, margin, reset_constant, ema):
    optimizer.zero_grad()

    model_output = model(x)
    logits = extract_logits(model_output)
    entropies = softmax_entropy(logits)

    # First filtering pass (reliability).
    idx1 = torch.where(entropies < margin)[0]
    if idx1.numel() == 0:
        return model_output, ema, False

    loss_first = entropies[idx1].mean(0)
    if torch.isnan(loss_first):
        return model_output, ema, False
    loss_first.backward()
    optimizer.first_step(zero_grad=True)

    # Second pass at perturbed weights.
    model_output_second = model(x)
    logits_second = extract_logits(model_output_second)
    entropies_second = softmax_entropy(logits_second)[idx1]
    idx2 = torch.where(entropies_second < margin)[0]
    if idx2.numel() == 0:
        optimizer.second_step(zero_grad=True)
        return model_output, ema, False

    loss_second = entropies_second[idx2].mean(0)
    if not torch.isnan(loss_second):
        ema = _update_ema(ema, float(loss_second.detach().item()))
        loss_second.backward()
    optimizer.second_step(zero_grad=True)

    need_reset = ema is not None and ema < reset_constant
    return model_output, ema, need_reset


def setup_sar(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    margin_e0: float = 0.4 * math.log(1000),
    reset_constant_em: float = 0.2,
    rho: float = 0.05,
    sam_adaptive: bool = False,
    base_optimizer_cls=torch.optim.SGD,
    base_optimizer_kwargs: dict | None = None,
) -> Tuple[SAR, list[str], str]:
    """Configure a model and wrap it with SAR."""
    base_optimizer_kwargs = base_optimizer_kwargs or {"momentum": 0.9}

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
    optimizer = SAM(
        params=params,
        base_optimizer=base_optimizer_cls,
        rho=rho,
        adaptive=sam_adaptive,
        lr=lr,
        **base_optimizer_kwargs,
    )
    sar = SAR(
        model=model,
        optimizer=optimizer,
        steps=steps,
        episodic=episodic,
        margin_e0=margin_e0,
        reset_constant_em=reset_constant_em,
    )
    return sar, selected_names, arch