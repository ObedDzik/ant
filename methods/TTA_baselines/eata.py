"""EATA baseline for test-time adaptation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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
    softmax_entropy,
)


def _update_model_probs(current_probs, new_probs):
    if current_probs is None:
        if new_probs.numel() == 0:
            return None
        with torch.no_grad():
            return new_probs.mean(0)

    if new_probs.numel() == 0:
        return current_probs
    with torch.no_grad():
        return 0.9 * current_probs + 0.1 * new_probs.mean(0)


class EATA(nn.Module):
    """Efficient anti-forgetting test-time adaptation.

    Reference: Niu et al., ICML 2022 — Efficient Test-Time Model Adaptation without Forgetting
    """

    def __init__(
        self,
        model,
        optimizer,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2000.0,
        steps: int = 1,
        episodic: bool = False,
        # Official default: math.log(1000) * 0.40
        e_margin: float = math.log(1000) * 0.40,
        d_margin: float = 0.05,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("EATA requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.steps = steps
        self.episodic = episodic
        self.e_margin = e_margin
        self.d_margin = d_margin

        # Sample-count trackers (matches official num_samples_update_1/2 diagnostics)
        self.num_samples_update_1 = 0  # samples passing entropy filter
        self.num_samples_update_2 = 0  # samples passing both filters

        self.current_model_probs = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        model_output = None
        for _ in range(self.steps):
            model_output, updated_probs, n1, n2 = forward_and_adapt_eata(
                x=x,
                model=self.model,
                optimizer=self.optimizer,
                fishers=self.fishers,
                e_margin=self.e_margin,
                current_model_probs=self.current_model_probs,
                fisher_alpha=self.fisher_alpha,
                d_margin=self.d_margin,
            )
            self.current_model_probs = updated_probs
            self.num_samples_update_1 += n1
            self.num_samples_update_2 += n2
        return model_output

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset EATA without saved states.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.current_model_probs = None

    def forward_no_adapt(self, x):
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.enable_grad()
def forward_and_adapt_eata(
    x,
    model,
    optimizer,
    fishers,
    e_margin,
    current_model_probs,
    # Note: official function-level default is 50.0; class-level default of 2000.0
    # matches the paper and official main.py. Use class-level value in practice.
    fisher_alpha: float = 2000.0,
    d_margin: float = 0.05,
):
    """Forward pass, filter samples, compute weighted entropy loss, and adapt.

    Returns:
        model_output: Output from the initial forward pass (pre-adaptation).
        updated_probs: Updated running mean probability vector.
        num_counts_1: Number of samples passing the entropy filter.
        num_counts_2: Number of samples passing both filters (used for update).
    """
    model_output = model(x)
    logits = extract_logits(model_output)
    entropies = softmax_entropy(logits)

    # Filter 1: remove unreliable (high-entropy) samples
    filter_ids_1 = torch.where(entropies < e_margin)[0]
    num_counts_1 = filter_ids_1.numel()

    selected_ent = entropies[filter_ids_1]
    selected_probs = (
        logits.softmax(1)[filter_ids_1]
        if filter_ids_1.numel() > 0
        else logits.new_zeros((0, logits.shape[1]))
    )

    # Filter 2: remove redundant samples (too similar to running mean)
    if current_model_probs is not None and filter_ids_1.numel() > 0:
        cosine_sim = F.cosine_similarity(
            current_model_probs.unsqueeze(0), selected_probs, dim=1
        )
        filter_ids_2 = torch.where(torch.abs(cosine_sim) < d_margin)[0]
        selected_ent = selected_ent[filter_ids_2]
        selected_probs = selected_probs[filter_ids_2]

    num_counts_2 = selected_probs.shape[0]

    # Update running mean from doubly-filtered probs
    updated_probs = _update_model_probs(current_model_probs, selected_probs)

    optimizer.zero_grad()

    if selected_ent.numel() > 0:
        # Reweight: samples closer to the margin get higher weight
        coeff = 1.0 / torch.exp(selected_ent.detach() - e_margin)
        loss = (selected_ent * coeff).mean(0)

        # EWC regularization to prevent forgetting
        if fishers is not None:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in fishers:
                    fisher_matrix, param_ref = fishers[name]
                    ewc_loss = ewc_loss + fisher_alpha * (
                        fisher_matrix * (param - param_ref) ** 2
                    ).sum()
            loss = loss + ewc_loss

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

    optimizer.zero_grad()
    return model_output, updated_probs, num_counts_1, num_counts_2


def compute_fishers_from_dict_loader(
    model: nn.Module,
    fisher_loader,
    device: torch.device,
    num_samples: int | None = None,
    use_true_labels: bool = True,
):
    """Estimate per-parameter Fisher information from a labelled loader.

    Accumulates squared gradients weighted by batch size, then normalises by
    total sample count — matching the official EATA implementation.

    Args:
        model: The source-pretrained model (should be in eval mode).
        fisher_loader: DataLoader yielding dicts with 'bmode' and optionally 'label'.
        device: Target device.
        num_samples: Stop after this many samples (None = full loader).
        use_true_labels: If True and 'label' is in the batch, use ground-truth
            labels for the cross-entropy loss (more accurate Fisher estimate).
            Falls back to pseudo-labels (argmax) when labels are unavailable.

    Returns:
        fishers: Dict mapping param name -> [fisher_diagonal, param_reference].
    """
    fishers: Dict[str, list] = {}
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_samples = 0

    model.eval()  # consistent with TTA inference mode

    for data in fisher_loader:
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        batch_size = data["bmode"].shape[0]

        model.zero_grad(set_to_none=True)

        model_output = model(data)
        logits = extract_logits(model_output)

        if use_true_labels and "label" in data:
            loss = loss_fn(logits, data["label"])
        else:
            # Fall back to pseudo-labels
            pseudo_targets = logits.argmax(dim=1).detach()
            loss = loss_fn(logits, pseudo_targets)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # Weight by batch size so final normalisation by total_samples is correct
            fisher_now = param.grad.detach().clone() ** 2
            if name in fishers:
                fishers[name][0] = fishers[name][0] + fisher_now * batch_size
            else:
                fishers[name] = [fisher_now * batch_size, param.detach().clone()]

        total_samples += batch_size
        if num_samples is not None and total_samples >= num_samples:
            break

    # Normalise by total sample count (not num_iters) to handle variable batch sizes
    if total_samples > 0:
        for name in fishers:
            fishers[name][0] = fishers[name][0] / float(total_samples)

    return fishers


def setup_eata(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    fishers=None,
    fisher_alpha: float = 2000.0,
    e_margin: float = math.log(1000) * 0.40,
    d_margin: float = 0.05,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[EATA, list[str], str]:
    """Configure a model and wrap it with EATA.

    Returns:
        (eata_model, adapted_param_names, inferred_architecture)
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
    eata = EATA(
        model=model,
        optimizer=optimizer,
        fishers=fishers,
        fisher_alpha=fisher_alpha,
        steps=steps,
        episodic=episodic,
        e_margin=e_margin,
        d_margin=d_margin,
    )
    return eata, selected_names, arch