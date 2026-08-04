"""RoTTA baseline for dynamic test-time adaptation.

Implementation follows the official RoTTA algorithm:
- Robust BatchNorm (RBN) adaptation layers
- Class-sensitive uncertainty-aware memory bank
- EMA teacher / student consistency with strong augmentations
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .common import (
    copy_model_and_optimizer,
    extract_logits,
    infer_architecture,
    load_model_and_optimizer,
)


@dataclass
class _MemoryItem:
    data: torch.Tensor
    uncertainty: float
    age: int = 0

    def increase_age(self):
        self.age += 1


class _CSTU:
    """Class-sensitive uncertainty memory bank used in RoTTA."""

    def __init__(self, capacity: int, num_class: int, lambda_t: float = 1.0, lambda_u: float = 1.0):
        if capacity <= 0:
            raise ValueError("RoTTA memory capacity must be positive.")
        if num_class <= 1:
            raise ValueError("RoTTA requires num_class > 1.")

        self.capacity = capacity
        self.num_class = num_class
        self.per_class = float(capacity) / float(num_class)
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data: list[list[_MemoryItem]] = [[] for _ in range(num_class)]

    def get_occupancy(self) -> int:
        return sum(len(v) for v in self.data)

    def per_class_dist(self) -> list[int]:
        return [len(v) for v in self.data]

    def get_majority_classes(self) -> list[int]:
        per_class = self.per_class_dist()
        max_occ = max(per_class)
        return [idx for idx, occ in enumerate(per_class) if occ == max_occ]

    def heuristic_score(self, age: int, uncertainty: float) -> float:
        time_score = 1.0 / (1.0 + math.exp(-float(age) / float(self.capacity)))
        unc_norm = uncertainty / math.log(float(self.num_class))
        return self.lambda_t * time_score + self.lambda_u * unc_norm

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def remove_from_classes(self, classes: list[int], score_base: float) -> bool:
        max_class = None
        max_index = None
        max_score = None

        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_class = cls
                    max_index = idx

        if max_class is None:
            return True

        if max_score is not None and max_score > score_base:
            self.data[max_class].pop(max_index)
            return True
        return False

    def remove_instance(self, cls: int, score: float) -> bool:
        class_occupied = len(self.data[cls])
        all_occupied = self.get_occupancy()

        if class_occupied < self.per_class:
            if all_occupied < self.capacity:
                return True
            return self.remove_from_classes(self.get_majority_classes(), score)

        return self.remove_from_classes([cls], score)

    def add_instance(self, instance: tuple[torch.Tensor, int, float]):
        x, prediction, uncertainty = instance
        new_score = self.heuristic_score(age=0, uncertainty=uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(
                _MemoryItem(data=x.detach().clone(), uncertainty=float(uncertainty), age=0)
            )
        self.add_age()

    def get_memory(self) -> tuple[list[torch.Tensor], list[float]]:
        memory_data: list[torch.Tensor] = []
        memory_age: list[float] = []
        for class_list in self.data:
            for item in class_list:
                memory_data.append(item.data)
                memory_age.append(float(item.age) / float(self.capacity))
        return memory_data, memory_age


class _MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.modules.batchnorm._BatchNorm, momentum: float):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.eps = bn_layer.eps

        if bn_layer.running_mean is not None and bn_layer.running_var is not None:
            source_mean = bn_layer.running_mean.detach().clone()
            source_var = bn_layer.running_var.detach().clone()
        else:
            source_mean = torch.zeros(self.num_features, device=bn_layer.weight.device)
            source_var = torch.ones(self.num_features, device=bn_layer.weight.device)

        self.register_buffer("source_mean", source_mean)
        self.register_buffer("source_var", source_var)
        self.weight = nn.Parameter(bn_layer.weight.detach().clone())
        self.bias = nn.Parameter(bn_layer.bias.detach().clone())


class _RobustBN1d(_MomentumBN):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)
            mean = (1.0 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1.0 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean = mean.detach().clone()
            self.source_var = var.detach().clone()
        else:
            mean, var = self.source_mean, self.source_var

        x = (x - mean.view(1, -1)) / torch.sqrt(var.view(1, -1) + self.eps)
        return x * self.weight.view(1, -1) + self.bias.view(1, -1)


class _RobustBN2d(_MomentumBN):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)
            mean = (1.0 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1.0 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean = mean.detach().clone()
            self.source_var = var.detach().clone()
        else:
            mean, var = self.source_mean, self.source_var

        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

def _replace_bn_with_rbn(model: nn.Module, bn_momentum: float) -> nn.Module:
    """Replace all BatchNorm1d/2d with RobustBN equivalents."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            rbn = _RobustBN2d(module, bn_momentum)
            setattr(model, name, rbn)
        elif isinstance(module, nn.BatchNorm1d):
            rbn = _RobustBN1d(module, bn_momentum)
            setattr(model, name, rbn)
        else:
            _replace_bn_with_rbn(module, bn_momentum)  # recurse into children
    return model

def _get_named_submodule(model: nn.Module, sub_name: str) -> nn.Module:
    module = model
    for part in sub_name.split("."):
        module = getattr(module, part)
    return module


def _set_named_submodule(model: nn.Module, sub_name: str, value: nn.Module):
    module = model
    names = sub_name.split(".")
    for idx, name in enumerate(names):
        if idx == len(names) - 1:
            setattr(module, name, value)
        else:
            module = getattr(module, name)


def _timeliness_reweighting(ages, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    age_tensor = torch.tensor(ages, dtype=dtype, device=device)
    return torch.exp(-age_tensor) / (1.0 + torch.exp(-age_tensor))


def _student_teacher_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)


class RoTTA(nn.Module):
    """Robust test-time adaptation in dynamic scenarios."""

    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        steps: int = 1,
        episodic: bool = False,
        memory_size: int = 64,
        update_frequency: int = 64,
        nu: float = 1e-3,
        lambda_t: float = 1.0,
        lambda_u: float = 1.0,
        bn_momentum: float = 0.05,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("RoTTA requires steps >= 1.")
        if update_frequency <= 0:
            raise ValueError("RoTTA update_frequency must be positive.")

        self.steps = steps
        self.episodic = episodic
        self.memory_size = memory_size
        self.update_frequency = update_frequency
        self.nu = nu
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.model = self._configure_model(model, bn_momentum)
        self.optimizer = optimizer
        self.model_ema = self._build_ema(self.model)

        self.mem: Optional[_CSTU] = None
        self.current_instance = 0

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

    def _configure_model(self, model: nn.Module, bn_momentum: float) -> nn.Module:
        model.train()
        model.requires_grad_(False)

        norm_names = []
        for name, submodule in model.named_modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
                norm_names.append(name)

        for name in norm_names:
            bn_layer = _get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                new_bn = _RobustBN1d(bn_layer, momentum=bn_momentum)
            else:
                new_bn = _RobustBN2d(bn_layer, momentum=bn_momentum)
            new_bn.requires_grad_(True)
            _set_named_submodule(model, name, new_bn)

        return model

    @staticmethod
    def _build_ema(model: nn.Module) -> nn.Module:
        ema_model = deepcopy(model)
        for p in ema_model.parameters():
            p.detach_()
        return ema_model

    @staticmethod
    def _update_ema_variables(ema_model: nn.Module, model: nn.Module, nu: float):
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data[:] = (1.0 - nu) * ema_p.data[:] + nu * p.data[:]

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
    #     aug_list = []
    #     for img in x_denorm:
    #         aug_img = self.strong_aug(img)
    #         noise = torch.randn_like(aug_img) * 0.005
    #         aug_list.append((aug_img + noise).clamp(0.0, 1.0))
    #     aug = torch.stack(aug_list, dim=0)
    #     return self._normalize(aug)

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

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        is_dict = isinstance(x, dict)
        batch_size = x['bmode'].shape[0] if is_dict else x.shape[0]

        with torch.no_grad():
            self.model.eval()
            self.model_ema.eval()
            ema_output = self.model_ema(x)
            ema_logits = extract_logits(ema_output)
            probs = torch.softmax(ema_logits, dim=1)
            pseudo_label = torch.argmax(probs, dim=1)
            entropy = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)

        if self.mem is None:
            self.mem = _CSTU(
                capacity=self.memory_size,
                num_class=int(ema_logits.shape[1]),
                lambda_t=self.lambda_t,
                lambda_u=self.lambda_u,
            )

        for i in range(batch_size):
            if is_dict:
                sample = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            else:
                sample = x[i]
            instance = (sample, int(pseudo_label[i].item()), float(entropy[i].item()))
            self.mem.add_instance(instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self._update_model()

        return ema_output
        
    def _update_model(self):
        if self.mem is None:
            return

        sup_data, ages = self.mem.get_memory()
        if len(sup_data) == 0:
            return

        self.model.train()
        self.model_ema.train()

        sup_batch = torch.stack(sup_data, dim=0)
        strong_batch = self._strong_augment(sup_batch)

        with torch.no_grad():
            ema_sup_out = self.model_ema(sup_batch)
            ema_logits = extract_logits(ema_sup_out)

        stu_sup_out = self.model(strong_batch)
        stu_logits = extract_logits(stu_sup_out)

        weights = _timeliness_reweighting(ages, device=stu_logits.device, dtype=stu_logits.dtype)
        loss = (_student_teacher_entropy(stu_logits, ema_logits) * weights).mean()

        if self.optimizer is not None and not torch.isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._update_ema_variables(self.model_ema, self.model, self.nu)

    def forward(self, x):
        if self.episodic:
            self.reset()

        model_output = None
        for _ in range(self.steps):
            model_output = self.forward_and_adapt(x)
        return model_output

    def forward_no_adapt(self, x):
        return self.model(x)

    def reset(self):
        if self.model_state is None:
            raise RuntimeError("Cannot reset RoTTA without saved model state.")

        if self.optimizer is not None and self.optimizer_state is not None:
            load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        else:
            self.model.load_state_dict(self.model_state, strict=True)

        self.model_ema.load_state_dict(self.ema_state, strict=True)
        self.mem = None
        self.current_instance = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _collect_rotta_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    params = []
    names = []
    for nm, module in model.named_modules():
        if isinstance(module, (_RobustBN1d, _RobustBN2d)):
            params.append(module.weight)
            params.append(module.bias)
            names.append(f"{nm}.weight" if nm else "weight")
            names.append(f"{nm}.bias" if nm else "bias")
    return params, names


def setup_rotta(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    memory_size: int = 64,
    update_frequency: int = 64,
    nu: float = 1e-3,
    lambda_t: float = 1.0,
    lambda_u: float = 1.0,
    bn_momentum: float = 0.05,
    mean=(0,0,0),
    std=(1,1,1),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[RoTTA, list[str], str]:
    """Configure a model and wrap it with RoTTA."""
    optimizer_kwargs = optimizer_kwargs or {}
    arch = infer_architecture(model)

    rotta = RoTTA(
        model=model,
        optimizer=None,
        steps=steps,
        episodic=episodic,
        memory_size=memory_size,
        update_frequency=update_frequency,
        nu=nu,
        lambda_t=lambda_t,
        lambda_u=lambda_u,
        bn_momentum=bn_momentum,
        mean=mean,
        std=std,
    )
    bn_count = sum(1 for m in rotta.model.modules() 
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    print(f"[RoTTA] Found {bn_count} BN layers to replace.")

    # Replace BN layers with RobustBN before collecting params
    _replace_bn_with_rbn(rotta.model, bn_momentum)

    params, names = _collect_rotta_params(rotta.model)
    if len(params) == 0:
        raise ValueError("RoTTA requires BatchNorm layers to replace with RobustBN.")

    rotta.optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
    rotta.model_state, rotta.optimizer_state = copy_model_and_optimizer(rotta.model, rotta.optimizer)
    rotta.ema_state = deepcopy(rotta.model_ema.state_dict())
    return rotta, names, arch