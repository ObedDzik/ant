"""PeTTA baseline for recurring test-time adaptation.

This implementation follows the original PeTTA design while remaining model-agnostic:
- robust BN adaptation layers
- class-sensitive sample memory
- EMA teacher/student updates
- anchor + regularization losses
- adaptive lambda/alpha with prototype divergence
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .common import (
    copy_model_and_optimizer,
    extract_logits,
    infer_architecture,
    load_model_and_optimizer,
)
from .rotta import _CSTU, _RobustBN1d, _RobustBN2d


def _softmax_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)


def _self_training(student_logits, student_logits_aug, teacher_logits):
    return (
        -0.25 * (teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)
        - 0.25 * (student_logits.softmax(1) * teacher_logits.log_softmax(1)).sum(1)
        - 0.25 * (teacher_logits.softmax(1) * student_logits_aug.log_softmax(1)).sum(1)
        - 0.25 * (student_logits_aug.softmax(1) * teacher_logits.log_softmax(1)).sum(1)
    )


class _DivergenceScore(nn.Module):
    """Computes source-target prototype divergence as in PeTTA."""

    def __init__(self, src_prototype: torch.Tensor, src_prototype_cov: torch.Tensor):
        super().__init__()
        self.src_proto = src_prototype
        self.src_proto_cov = src_prototype_cov

    def forward(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor) -> torch.Tensor:
        lbl_uniq = torch.unique(pseudo_lbls)
        group_avgs = []
        for lbl in lbl_uniq:
            group_avgs.append(feats[pseudo_lbls == lbl].mean(dim=0, keepdim=True))
        pred_proto = torch.cat(group_avgs, dim=0)
        target_proto = self.src_proto[lbl_uniq]
        target_cov = self.src_proto_cov[lbl_uniq]
        return ((pred_proto - target_proto).pow(2) / (target_cov + 1e-6)).mean()


class _PrototypeMemory:
    def __init__(self, src_prototype: torch.Tensor):
        self.src_proto = src_prototype.detach().clone()
        self.mem_proto = src_prototype.detach().clone()

    def update(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor, nu: float = 0.05):
        lbl_uniq = torch.unique(pseudo_lbls)
        for lbl in lbl_uniq:
            batch_avg = feats[pseudo_lbls == lbl].mean(dim=0)
            self.mem_proto[lbl] = (1.0 - nu) * self.mem_proto[lbl] + nu * batch_avg


class PeTTA(nn.Module):
    """Persistent test-time adaptation with prototype-aware updates."""

    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        steps: int = 1,
        episodic: bool = False,
        memory_size: int = 64,
        lambda_t: float = 1.0,
        lambda_u: float = 1.0,
        alpha_0: float = 1e-3,
        lambda_0: float = 10.0,
        al_wgt: float = 1.0,
        regularizer: str = "cosine",
        loss_func: str = "sce",
        adaptive_lambda: bool = True,
        adaptive_alpha: bool = True,
        proto_nu: float = 0.05,
        bn_momentum: float = 0.05,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        source_prototypes: Optional[torch.Tensor] = None,
        source_covariances: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("PeTTA requires steps >= 1.")
        if regularizer not in {"l2", "cosine", "none"}:
            raise ValueError("regularizer must be one of {'l2','cosine','none'}.")
        if loss_func not in {"sce", "ce"}:
            raise ValueError("loss_func must be one of {'sce','ce'}.")

        self.steps = steps
        self.episodic = episodic
        self.memory_size = memory_size
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.alpha_0 = alpha_0
        self.alpha = alpha_0
        self.lambda_0 = lambda_0
        self.al_wgt = al_wgt
        self.regularizer = regularizer
        self.loss_func = loss_func
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_alpha = adaptive_alpha
        self.proto_nu = proto_nu
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.num_classes = num_classes

        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

        self.model = self._configure_model(model, bn_momentum)
        self.optimizer = optimizer
        self.model_ema = self._build_ema(self.model)
        self.model_init = self._build_ema(self.model)

        self.sample_mem: Optional[_CSTU] = None

        self.src_prototypes = source_prototypes.detach().clone() if source_prototypes is not None else None
        self.src_covariances = source_covariances.detach().clone() if source_covariances is not None else None
        self.proto_mem: Optional[_PrototypeMemory] = None
        self.divg_score: Optional[_DivergenceScore] = None

        self.strong_aug = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=(0.6, 1.4),
                    contrast=(0.7, 1.3),
                    saturation=(0.5, 1.5),
                    hue=(-0.06, 0.06),
                ),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(1.0 / 16.0, 1.0 / 16.0),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.001, 0.5)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
        self.ema_state = deepcopy(self.model_ema.state_dict())
        self.init_state = deepcopy(self.model_init.state_dict())

    def _configure_model(self, model: nn.Module, bn_momentum: float) -> nn.Module:
        model.train()
        model.requires_grad_(False)

        bn_names = []
        for name, submodule in model.named_modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_names.append(name)

        for name in bn_names:
            layer = self._get_submodule(model, name)
            if isinstance(layer, nn.BatchNorm1d):
                new_bn = _RobustBN1d(layer, momentum=bn_momentum)
            else:
                new_bn = _RobustBN2d(layer, momentum=bn_momentum)
            new_bn.requires_grad_(True)
            self._set_submodule(model, name, new_bn)

        return model

    @staticmethod
    def _get_submodule(model: nn.Module, path: str) -> nn.Module:
        module = model
        for token in path.split("."):
            module = getattr(module, token)
        return module

    @staticmethod
    def _set_submodule(model: nn.Module, path: str, value: nn.Module):
        module = model
        tokens = path.split(".")
        for idx, token in enumerate(tokens):
            if idx == len(tokens) - 1:
                setattr(module, token, value)
            else:
                module = getattr(module, token)

    @staticmethod
    def _build_ema(model: nn.Module) -> nn.Module:
        ema = deepcopy(model)
        for p in ema.parameters():
            p.detach_()
        return ema

    @staticmethod
    def _update_ema_variables(ema_model: nn.Module, model: nn.Module, alpha: float):
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data[:] = (1.0 - alpha) * ema_p.data[:] + alpha * p.data[:]

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(1, -1, 1, 1)
        return tensor * std + mean

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(1, -1, 1, 1)
        return (tensor - mean) / std

    # def _strong_augment(self, x: torch.Tensor) -> torch.Tensor:
    #     x_denorm = self._denormalize(x).clamp(0.0, 1.0)
    #     out = []
    #     for img in x_denorm:
    #         aug_img = self.strong_aug(img)
    #         noise = torch.randn_like(aug_img) * 0.005
    #         out.append((aug_img + noise).clamp(0.0, 1.0))
    #     return self._normalize(torch.stack(out, dim=0))

    def _strong_augment(self, x: dict) -> dict:
        """Dict-aware version of _strong_augment, operates only on 'bmode'."""
        bmode = x['bmode']  # (B, C, H, W)
        x_denorm = self._denormalize(bmode).clamp(0.0, 1.0)
        out = []
        for img in x_denorm:
            aug_img = self.strong_aug(img)
            noise = torch.randn_like(aug_img) * 0.005
            out.append((aug_img + noise).clamp(0.0, 1.0))
        
        aug_x = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        aug_x['bmode'] = self._normalize(torch.stack(out, dim=0))
        return aug_x

    def _forward_features_and_logits(self, model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_extractor is None or self.classifier_head is None:
            out = model(x)
            logits = extract_logits(out)
            return logits, logits

        feats = self.feature_extractor(model, x)
        logits = self.classifier_head(model, feats)
        if not isinstance(feats, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise TypeError("feature_extractor and classifier_head must return torch.Tensor.")
        return feats, logits

    def _bootstrap_source_stats(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor):
        device = feats.device
        dtype = feats.dtype
        if self.num_classes is None:
            self.num_classes = int(pseudo_lbls.max().item() + 1)

        if self.src_prototypes is None:
            global_mean = feats.mean(dim=0, keepdim=True)
            src_proto = global_mean.repeat(self.num_classes, 1)
            for cls in range(self.num_classes):
                mask = pseudo_lbls == cls
                if mask.any():
                    src_proto[cls] = feats[mask].mean(dim=0)
            self.src_prototypes = src_proto.detach().clone()

        if self.src_covariances is None:
            feat_var = feats.var(dim=0, unbiased=False, keepdim=True) + 1e-6
            self.src_covariances = feat_var.repeat(self.num_classes, 1).detach().clone()

        self.src_prototypes = self.src_prototypes.to(device=device, dtype=dtype)
        self.src_covariances = self.src_covariances.to(device=device, dtype=dtype)

        if self.proto_mem is None:
            self.proto_mem = _PrototypeMemory(self.src_prototypes)
        if self.divg_score is None:
            self.divg_score = _DivergenceScore(self.src_prototypes, self.src_covariances).to(device)

    def _regularization_loss(self, model: nn.Module) -> torch.Tensor:
        if self.regularizer == "none":
            dev = next(model.parameters()).device
            return torch.tensor(0.0, device=dev)

        reg = None
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ref = self.init_state[name].to(param.device)
            if self.regularizer == "l2":
                term = (param - ref).pow(2).sum()
            else:
                term = -torch.cosine_similarity(param.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).mean()

            reg = term if reg is None else reg + term
            count += 1

        if reg is None or count == 0:
            dev = next(model.parameters()).device
            return torch.tensor(0.0, device=dev)
        return reg / float(count)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        ema_output = None
        
        # Helper to get batch size regardless of input type
        is_dict = isinstance(x, dict)
        batch_size = x['bmode'].shape[0] if is_dict else x.shape[0]

        with torch.no_grad():
            self.model_ema.eval()
            if self.feature_extractor is None or self.classifier_head is None:
                ema_output = self.model_ema(x)
                ema_logits = extract_logits(ema_output)
                ema_feat = ema_logits
            else:
                ema_feat, ema_logits = self._forward_features_and_logits(self.model_ema, x)
                ema_output = ema_logits
            probs = torch.softmax(ema_logits, dim=1)
            pseudo_lbls = probs.argmax(dim=1)
            entropy = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)

        if self.num_classes is None:
            self.num_classes = int(ema_logits.shape[1])

        if self.sample_mem is None:
            self.sample_mem = _CSTU(
                capacity=self.memory_size,
                num_class=self.num_classes,
                lambda_t=self.lambda_t,
                lambda_u=self.lambda_u,
            )

        # Store individual samples — slice dict or tensor uniformly
        for idx in range(batch_size):
            if is_dict:
                sample = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            else:
                sample = x[idx]
            self.sample_mem.add_instance((sample, int(pseudo_lbls[idx].item()), float(entropy[idx].item())))

        sup_data, _ = self.sample_mem.get_memory()
        if len(sup_data) == 0:
            return ema_logits

        # Reconstruct batched input from memory samples
        if isinstance(sup_data[0], dict):
            sup_batch = {}
            for k in sup_data[0].keys():
                vals = [s[k] for s in sup_data]
                sup_batch[k] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
            strong_sup = self._strong_augment_dict(sup_batch)
        else:
            sup_data = torch.stack(sup_data, dim=0)
            strong_sup = self._strong_augment(sup_data)

        self.model.train()
        self.model_ema.train()
        self.model_init.train()

        ema_sup_feat, ema_sup_logits = self._forward_features_and_logits(self.model_ema, sup_data)
        sup_pseudo_lbls = ema_sup_logits.argmax(dim=1)
        _, p_ori = self._forward_features_and_logits(self.model, sup_data)
        _, init_logits = self._forward_features_and_logits(self.model_init, sup_data)
        _, p_aug = self._forward_features_and_logits(self.model, strong_sup)

        if self.loss_func == "sce":
            cls_lss = _self_training(p_ori, p_aug, ema_sup_logits).mean()
        else:
            cls_lss = _softmax_entropy(p_aug, ema_sup_logits).mean()

        reg_lss = self._regularization_loss(self.model)
        anchor_lss = _softmax_entropy(p_aug, init_logits).mean()

        self._bootstrap_source_stats(ema_sup_feat.detach(), sup_pseudo_lbls)
        reg_wgt = self.lambda_0

        if self.adaptive_lambda or self.adaptive_alpha:
            lbl_uniq = torch.unique(sup_pseudo_lbls)
            divg_scr = 1.0 - torch.exp(
                -self.divg_score(self.proto_mem.mem_proto[lbl_uniq], lbl_uniq)
            )
            self.proto_mem.update(feats=ema_sup_feat.detach(), pseudo_lbls=sup_pseudo_lbls, nu=self.proto_nu)

            if self.adaptive_lambda:
                reg_wgt = divg_scr * self.lambda_0
            if self.adaptive_alpha:
                self.alpha = (1.0 - divg_scr) * self.alpha_0

        total_lss = cls_lss + reg_wgt * reg_lss + self.al_wgt * anchor_lss

        if self.optimizer is not None and not torch.isnan(total_lss):
            self.optimizer.zero_grad()
            total_lss.backward()
            self.optimizer.step()

        self._update_ema_variables(self.model_ema, self.model, float(self.alpha))
        return ema_output

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
        if self.model_state is None:
            raise RuntimeError("Cannot reset PeTTA without saved model state.")

        if self.optimizer is not None and self.optimizer_state is not None:
            load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        else:
            self.model.load_state_dict(self.model_state, strict=True)

        self.model_ema.load_state_dict(self.ema_state, strict=True)
        self.model_init.load_state_dict(self.init_state, strict=True)
        self.sample_mem = None
        self.proto_mem = None
        self.divg_score = None
        self.alpha = self.alpha_0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _collect_petta_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    params = []
    names = []
    for nm, module in model.named_modules():
        if isinstance(module, (_RobustBN1d, _RobustBN2d)):
            params.append(module.weight)
            params.append(module.bias)
            names.append(f"{nm}.weight" if nm else "weight")
            names.append(f"{nm}.bias" if nm else "bias")
    return params, names


@torch.no_grad()
def compute_source_prototypes(
    model: nn.Module,
    source_loader,
    num_classes: int,
    feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute class-wise source feature means and diagonal covariances.

    The loader must yield `(images, labels, ...)` where labels are integer class ids.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    feats_by_class = [[] for _ in range(num_classes)]
    seen = 0

    for batch in source_loader:
        if isinstance(batch, dict):
            images = batch['bmode']
            labels = batch['label']  # adjust key to match your dataset
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("source_loader must return (images, labels, ...) or a dict.")

        images = images.to(device)
        labels = labels.to(device)

        if feature_extractor is None or classifier_head is None:
            out = model(batch if isinstance(batch, dict) else images)
            feats = extract_logits(out)
        else:
            feats = feature_extractor(model, batch if isinstance(batch, dict) else images)

        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                feats_by_class[cls].append(feats[mask].detach().cpu())

        seen += int(images.shape[0])
        if max_samples is not None and seen >= max_samples:
            break

    feat_dim = None
    for cls_feats in feats_by_class:
        if len(cls_feats) > 0:
            feat_dim = cls_feats[0].shape[1]
            break
    if feat_dim is None:
        raise RuntimeError("Could not compute source prototypes: no labeled features collected.")

    means = []
    covs = []
    global_pool = torch.cat([f for cls_feats in feats_by_class for f in cls_feats], dim=0)
    global_mean = global_pool.mean(dim=0, keepdim=True)
    global_cov = global_pool.var(dim=0, unbiased=False, keepdim=True) + 1e-6

    for cls in range(num_classes):
        if len(feats_by_class[cls]) == 0:
            means.append(global_mean)
            covs.append(global_cov)
            continue

        cls_feats = torch.cat(feats_by_class[cls], dim=0)
        means.append(cls_feats.mean(dim=0, keepdim=True))
        covs.append(cls_feats.var(dim=0, unbiased=False, keepdim=True) + 1e-6)

    return torch.cat(means, dim=0), torch.cat(covs, dim=0)


def setup_petta(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    memory_size: int = 64,
    lambda_t: float = 1.0,
    lambda_u: float = 1.0,
    alpha_0: float = 1e-3,
    lambda_0: float = 10.0,
    al_wgt: float = 1.0,
    regularizer: str = "cosine",
    loss_func: str = "sce",
    adaptive_lambda: bool = True,
    adaptive_alpha: bool = True,
    proto_nu: float = 0.05,
    bn_momentum: float = 0.05,
    mean=(0, 0, 0),
    std=(1, 1, 1),
    source_prototypes: Optional[torch.Tensor] = None,
    source_covariances: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[PeTTA, list[str], str]:
    """Configure a model and wrap it with PeTTA."""
    optimizer_kwargs = optimizer_kwargs or {}
    arch = infer_architecture(model)

    petta = PeTTA(
        model=model,
        optimizer=None,
        steps=steps,
        episodic=episodic,
        memory_size=memory_size,
        lambda_t=lambda_t,
        lambda_u=lambda_u,
        alpha_0=alpha_0,
        lambda_0=lambda_0,
        al_wgt=al_wgt,
        regularizer=regularizer,
        loss_func=loss_func,
        adaptive_lambda=adaptive_lambda,
        adaptive_alpha=adaptive_alpha,
        proto_nu=proto_nu,
        bn_momentum=bn_momentum,
        mean=mean,
        std=std,
        source_prototypes=source_prototypes,
        source_covariances=source_covariances,
        num_classes=num_classes,
        feature_extractor=feature_extractor,
        classifier_head=classifier_head,
    )

    params, names = _collect_petta_params(petta.model)
    if len(params) == 0:
        raise ValueError("PeTTA requires BatchNorm layers to replace with RobustBN.")

    petta.optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
    petta.model_state, petta.optimizer_state = copy_model_and_optimizer(petta.model, petta.optimizer)
    petta.ema_state = deepcopy(petta.model_ema.state_dict())
    petta.init_state = deepcopy(petta.model_init.state_dict())
    return petta, names, arch