"""RMT baseline for source-aware test-time adaptation.

Based on the original RMT implementation from test-time-adaptation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .common import extract_logits, infer_architecture, load_model_and_optimizer


def _replace_logits(model_output, new_logits: torch.Tensor):
    if isinstance(model_output, torch.Tensor):
        return new_logits
    if isinstance(model_output, tuple) and len(model_output) > 0:
        return (new_logits,) + tuple(model_output[1:])
    if isinstance(model_output, list) and len(model_output) > 0:
        return [new_logits] + list(model_output[1:])
    return new_logits


def _symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return -(1.0 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (
        x.softmax(1) * x_ema.log_softmax(1)
    ).sum(1)


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
            transforms.Pad(padding=int(image_size / 2), padding_mode="edge"),
            transforms.RandomAffine(
                degrees=15,
                translate=(1.0 / 16.0, 1.0 / 16.0),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.001, 0.5)),
            transforms.CenterCrop(size=image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda t: torch.clip(t, 0.0, 1.0)),
        ]
    )


def _configure_model_for_rmt(model: nn.Module):
    model.eval()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif isinstance(module, nn.BatchNorm1d):
            module.train()
            module.requires_grad_(True)
        else:
            module.requires_grad_(True)
    return model


def _collect_trainable_params(model: nn.Module):
    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            names.append(name)
    return params, names


def _ema_update_model(model_to_update: nn.Module, model_to_merge: nn.Module, momentum: float, update_all: bool = True):
    if momentum >= 1.0:
        return
    with torch.no_grad():
        for p_update, p_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if p_update.requires_grad or update_all:
                p_update.data = momentum * p_update.data + (1.0 - momentum) * p_merge.data.to(p_update.device)


@torch.no_grad()
def compute_rmt_source_prototypes(
    model: nn.Module,
    source_loader,
    num_classes: int,
    feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    max_samples: int = 100000,
) -> torch.Tensor:
    """Compute class-wise source prototypes for RMT.

    The loader must yield `(images, labels, ...)` with integer labels.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    feat_sums = None
    feat_counts = torch.zeros(num_classes, dtype=torch.long)
    total = 0

    for batch in source_loader:
        # if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        #     raise ValueError("source_loader must yield (images, labels, ...).")
        images, labels = batch['bmode'].to(device), batch['label']
        labels = labels.long().cpu()

        if feature_extractor is None or classifier_head is None:
            logits = extract_logits(model(images))
            feats = logits
        else:
            feats = feature_extractor(model, images)
            if not isinstance(feats, torch.Tensor):
                raise TypeError("feature_extractor must return a torch.Tensor.")

        feats_cpu = feats.detach().cpu()

        if feat_sums is None:
            feat_sums = torch.zeros(num_classes, feats_cpu.shape[1], dtype=feats_cpu.dtype)

        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                feat_sums[cls] += feats_cpu[mask].sum(dim=0)
                feat_counts[cls] += int(mask.sum().item())

        total += int(images.shape[0])
        if total >= max_samples:
            break

    if feat_sums is None:
        raise RuntimeError("Could not compute source prototypes: no source samples processed.")

    global_mean = feat_sums.sum(dim=0, keepdim=True) / max(int(feat_counts.sum().item()), 1)
    prototypes = torch.zeros_like(feat_sums)
    for cls in range(num_classes):
        if feat_counts[cls] > 0:
            prototypes[cls] = feat_sums[cls] / float(feat_counts[cls].item())
        else:
            prototypes[cls] = global_mean.squeeze(0)
    return prototypes


class RMT(nn.Module):
    """RMT adaptation with self-training + contrastive source prototypes."""

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        source_loader=None,
        source_prototypes: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        image_size: int = 224,
        lambda_ce_src: float = 1.0,
        lambda_ce_trg: float = 1.0,
        lambda_cont: float = 1.0,
        teacher_momentum: float = 0.999,
        temperature: float = 0.1,
        contrast_mode: str = "all",
        projection_dim: int = 128,
        warmup_steps: int = 0,
        final_lr: float = 1e-3,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("RMT requires steps >= 1.")
        if contrast_mode not in {"one", "all"}:
            raise ValueError("contrast_mode must be 'one' or 'all'.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.source_loader = source_loader
        self.source_loader_iter = iter(source_loader) if source_loader is not None else None

        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

        self.image_size = image_size
        self.lambda_ce_src = lambda_ce_src
        self.lambda_ce_trg = lambda_ce_trg
        self.lambda_cont = lambda_cont
        self.teacher_momentum = teacher_momentum
        self.temperature = temperature
        self.base_temperature = temperature
        self.contrast_mode = contrast_mode
        self.projection_dim = projection_dim
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr

        self.tta_transform = _build_tta_transform(image_size=image_size)
        self.model_ema = deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.detach_()

        self.num_classes = num_classes
        self.source_prototypes = source_prototypes.detach().clone() if source_prototypes is not None else None

        feat_dim = None
        if self.source_prototypes is not None:
            feat_dim = int(self.source_prototypes.shape[1])
        elif self.num_classes is not None and self.feature_extractor is None and self.classifier_head is None:
            feat_dim = int(self.num_classes)

        if feat_dim is not None and self.projection_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim),
            ).to(next(self.model.parameters()).device)
            self.optimizer.add_param_group({"params": self.projector.parameters(), "lr": self.optimizer.param_groups[0]["lr"]})
        else:
            self.projector = nn.Identity()

        if self.warmup_steps > 0 and self.source_loader is not None:
            self._warmup()

        self.models = [self.model, self.model_ema, self.projector]
        self.model_states = [deepcopy(m.state_dict()) for m in self.models]
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def _forward_features_logits(self, model: nn.Module, x: torch.Tensor):
        if self.feature_extractor is None or self.classifier_head is None:
            model_output = model(x)
            logits = extract_logits(model_output)
            features = logits
            return model_output, features, logits

        features = self.feature_extractor(model, x)
        logits = self.classifier_head(model, features)
        if not isinstance(features, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise TypeError("feature_extractor and classifier_head must return torch.Tensor.")
        return logits, features, logits

    def _ensure_source_prototypes(self, features: torch.Tensor, logits: torch.Tensor):
        if self.source_prototypes is not None:
            self.source_prototypes = self.source_prototypes.to(device=features.device, dtype=features.dtype)
            return

        if self.num_classes is None:
            self.num_classes = int(logits.shape[1])
        preds = logits.argmax(dim=1)

        global_mean = features.mean(dim=0, keepdim=True)
        prototypes = global_mean.repeat(self.num_classes, 1)
        for cls in range(self.num_classes):
            mask = preds == cls
            if mask.any():
                prototypes[cls] = features[mask].mean(dim=0)
        self.source_prototypes = prototypes.detach().clone()

    def _contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        device = features.device

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask.")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num labels does not match number of features.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == "one":
            anchor_feature = self.projector(features[:, 0])
            anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
            anchor_count = 1
        else:
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1e-12)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def _next_source_batch(self):
        if self.source_loader is None:
            return None
        try:
            return next(self.source_loader_iter)
        except StopIteration:
            self.source_loader_iter = iter(self.source_loader)
            return next(self.source_loader_iter)

    @torch.enable_grad()
    def _warmup(self):
        if self.source_loader is None or self.warmup_steps <= 0:
            return

        for idx in range(self.warmup_steps):
            scale = float(idx + 1) / float(self.warmup_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = self.final_lr * scale

            batch = self._next_source_batch()
            if batch is None:
                break

            imgs_src = batch[0].to(next(self.model.parameters()).device)
            out = self.model(imgs_src)
            logits = extract_logits(out)
            out_ema = self.model_ema(imgs_src)
            logits_ema = extract_logits(out_ema)

            loss = _symmetric_cross_entropy(logits, logits_ema).mean(0)
            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            _ema_update_model(
                model_to_update=self.model_ema,
                model_to_merge=self.model,
                momentum=self.teacher_momentum,
                update_all=True,
            )

        for group in self.optimizer.param_groups:
            group["lr"] = self.final_lr

    def _loss_calculation(self, x):
        model_output_test, features_test, logits_test = self._forward_features_logits(self.model, x)
        _, features_aug_test, logits_aug_test = self._forward_features_logits(self.model, self.tta_transform(x))

        with torch.no_grad():
            out_ema = self.model_ema(x)
            logits_ema = extract_logits(out_ema)

        self._ensure_source_prototypes(features_test, logits_test)

        src_proto = self.source_prototypes.to(device=features_test.device, dtype=features_test.dtype)
        src_norm = F.normalize(src_proto, dim=1)
        feat_norm = F.normalize(features_test, dim=1)
        sim = torch.matmul(src_norm, feat_norm.T)
        nearest_idx = sim.argmax(dim=0)

        triplet_features = torch.cat(
            [
                src_proto[nearest_idx].unsqueeze(1),
                features_test.unsqueeze(1),
                features_aug_test.unsqueeze(1),
            ],
            dim=1,
        )
        loss_contrastive = self._contrastive_loss(features=triplet_features, labels=None)

        loss_self_training = (
            0.5 * _symmetric_cross_entropy(logits_test, logits_ema)
            + 0.5 * _symmetric_cross_entropy(logits_aug_test, logits_ema)
        ).mean(0)

        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive

        if self.lambda_ce_src > 0.0 and self.source_loader is not None:
            src_batch = self._next_source_batch()
            if src_batch is not None and len(src_batch) >= 2:
                imgs_src = src_batch[0].to(features_test.device)
                labels_src = src_batch[1].to(features_test.device).long()
                _, _, logits_src = self._forward_features_logits(self.model, imgs_src)
                loss = loss + self.lambda_ce_src * F.cross_entropy(logits_src, labels_src)

        ensemble_logits = logits_test + logits_ema
        outputs = _replace_logits(model_output_test, ensemble_logits)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs, loss = self._loss_calculation(x)
        if not torch.isnan(loss):
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        _ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.teacher_momentum,
            update_all=True,
        )
        return outputs

    def forward(self, x):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x)
        return out

    def forward_no_adapt(self, x):
        out_test = self.model(x)
        out_ema = self.model_ema(x)
        logits = extract_logits(out_test) + extract_logits(out_ema)
        return _replace_logits(out_test, logits)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset RMT without saved states.")

        for model, state in zip(self.models, self.model_states):
            model.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.source_loader_iter = iter(self.source_loader) if self.source_loader is not None else None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_rmt(
    model: nn.Module,
    source_loader=None,
    num_classes: int | None = None,
    lr: float = 1e-2,
    steps: int = 1,
    episodic: bool = False,
    image_size: int = 224,
    lambda_ce_src: float = 1.0,
    lambda_ce_trg: float = 1.0,
    lambda_cont: float = 1.0,
    teacher_momentum: float = 0.999,
    temperature: float = 0.1,
    contrast_mode: str = "all",
    projection_dim: int = 128,
    warmup_samples: int = 0,
    source_prototypes: Optional[torch.Tensor] = None,
    feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    classifier_head: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs: dict | None = None,
) -> Tuple[RMT, list[str], str]:
    """Configure a model and wrap it with RMT."""
    optimizer_kwargs = optimizer_kwargs or {"momentum": 0.9}
    arch = infer_architecture(model)

    model = _configure_model_for_rmt(model)
    params, names = _collect_trainable_params(model)
    if len(params) == 0:
        raise ValueError("RMT found no trainable parameters after configuration.")

    if source_prototypes is None and source_loader is not None and num_classes is not None:
        device = next(model.parameters()).device
        source_prototypes = compute_rmt_source_prototypes(
            model=model,
            source_loader=source_loader,
            num_classes=num_classes,
            feature_extractor=feature_extractor,
            classifier_head=classifier_head,
            device=device,
        )

    optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)

    source_bs = getattr(source_loader, "batch_size", 1) if source_loader is not None else 1
    warmup_steps = int(warmup_samples // max(source_bs, 1)) if warmup_samples > 0 else 0

    rmt = RMT(
        model=model,
        optimizer=optimizer,
        steps=steps,
        episodic=episodic,
        source_loader=source_loader,
        source_prototypes=source_prototypes,
        num_classes=num_classes,
        feature_extractor=feature_extractor,
        classifier_head=classifier_head,
        image_size=image_size,
        lambda_ce_src=lambda_ce_src,
        lambda_ce_trg=lambda_ce_trg,
        lambda_cont=lambda_cont,
        teacher_momentum=teacher_momentum,
        temperature=temperature,
        contrast_mode=contrast_mode,
        projection_dim=projection_dim,
        warmup_steps=warmup_steps,
        final_lr=lr,
    )
    return rmt, names, arch