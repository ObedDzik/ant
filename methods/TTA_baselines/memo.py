"""MEMO baseline for test-time adaptation."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    load_model_and_optimizer,
)


def marginal_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of marginal prediction over a batch of augmented views."""
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    avg_log_probs = log_probs.logsumexp(dim=0) - np.log(log_probs.shape[0])
    avg_log_probs = torch.clamp(avg_log_probs, min=torch.finfo(avg_log_probs.dtype).min)
    return -(avg_log_probs * torch.exp(avg_log_probs)).sum(dim=-1)


def _int_parameter(level, maxval):
    return int(level * maxval / 10)


def _rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


def _autocontrast(img, level=None):
    return ImageOps.autocontrast(img)


def _equalize(img, level=None):
    return ImageOps.equalize(img)


def _rotate(img, level):
    degrees = _int_parameter(_rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)


def _solarize(img, level):
    level = _int_parameter(_rand_lvl(level), 256)
    return ImageOps.solarize(img, 256 - level)


def _posterize(img, level):
    level = _int_parameter(_rand_lvl(level), 4)
    return ImageOps.posterize(img, 4 - level)


def create_augmix_augmentation(image_size: int = 224) -> Callable[[Image.Image], Image.Image]:
    augmentations = [
        _autocontrast,
        _equalize,
        lambda x: _rotate(x, 1),
        lambda x: _solarize(x, 1),
        lambda x: _posterize(x, 1),
    ]

    def augmix_fn(pil_img: Image.Image) -> Image.Image:
        if np.random.rand() > 0.5:
            pil_img = transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))(pil_img)
        else:
            pil_img = transforms.Resize(int(image_size * 256 / 224))(pil_img)
            pil_img = transforms.RandomCrop(image_size)(pil_img)

        if np.random.rand() > 0.5:
            pil_img = transforms.RandomHorizontalFlip(p=1.0)(pil_img)

        depth = np.random.randint(1, 4)
        for _ in range(depth):
            pil_img = np.random.choice(augmentations)(pil_img)
        return pil_img

    return augmix_fn


def create_standard_augmentation(image_size: int = 224) -> Callable[[Image.Image], Image.Image]:
    aug_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ]
    )

    def standard_fn(pil_img: Image.Image) -> Image.Image:
        return aug_transform(pil_img)

    return standard_fn


class MEMO(nn.Module):
    """Adapt model per-sample with marginal entropy minimization over augmentations."""

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        batch_size: int = 32,
        episodic: bool = True,
        augmentation_type: str = "augmix",
        image_size: int = 224,
        mean=(0, 0, 0),
        std=(1, 1, 1),
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("MEMO requires steps >= 1.")
        if batch_size <= 0:
            raise ValueError("MEMO batch_size must be positive.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.batch_size = batch_size
        self.episodic = episodic
        self.mean = tuple(mean)
        self.std = tuple(std)

        if augmentation_type == "augmix":
            self.aug_fn = create_augmix_augmentation(image_size=image_size)
        else:
            self.aug_fn = create_standard_augmentation(image_size=image_size)

        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        is_dict = isinstance(x, dict)
        batch_size = x['bmode'].shape[0] if is_dict else x.shape[0]

        all_outputs = []

        for i in range(batch_size):
            # Slice out a single sample
            if is_dict:
                x_single = {}
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x_single[k] = v[i].unsqueeze(0)
                    elif isinstance(v, list):
                        # lists of strings/scalars — slice the i-th element, wrap in list
                        x_single[k] = [v[i]]
                    else:
                        x_single[k] = v
            else:
                x_single = x[i].unsqueeze(0)  # (1, C, H, W)

            # Episodic reset per sample so each adapts from a clean state
            if self.episodic:
                self.reset()

            with torch.enable_grad():
                self.model.train()
                for _ in range(self.steps):
                    aug_x = self._generate_augmented_views(x_single)
                    self.optimizer.zero_grad()
                    aug_out = self.model(aug_x)
                    aug_logits = extract_logits(aug_out)
                    loss = marginal_entropy(aug_logits)
                    loss.backward()
                    self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                out = self.model(x_single)
                # print("[MEMO single fwd keys]:", out.keys() if isinstance(out, dict) else type(out))

                all_outputs.append(out)# keep the raw dict output, not just logits

        # Reconstruct a batched dict from the list of single-sample output dicts
        if isinstance(all_outputs[0], dict):
            batched_output = {}
            for k in all_outputs[0].keys():
                vals = [o[k] for o in all_outputs]
                if isinstance(vals[0], torch.Tensor):
                    batched_output[k] = torch.cat(vals, dim=0)
                elif isinstance(vals[0], list):
                    # Check if it's a list of tensors or a list of strings
                    if len(vals[0]) > 0 and isinstance(vals[0][0], torch.Tensor):
                        # e.g. image_level_classification_outputs: list of tensors
                        batched_output[k] = [
                            torch.cat([v[i] for v in vals], dim=0)
                            for i in range(len(vals[0]))
                        ]
                    else:
                        # list of strings or other non-tensor — flatten across samples
                        batched_output[k] = [item for v in vals for item in v]
                else:
                    batched_output[k] = vals[0]
            return batched_output
        else:
            return torch.cat(all_outputs, dim=0)


    def forward_no_adapt(self, x):
        return self.model(x)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset MEMO without saved states.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
        return tensor * std + mean

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
        return (tensor - mean) / std

    def _generate_augmented_views(self, x) -> dict:
        """
        Supports both plain Tensor and dict inputs.
        For dict inputs, augments only the 'bmode' key and stacks
        all augmented views into a single batched dict.
        """
        if isinstance(x, dict):
            bmode = x['bmode']  # shape: (1, C, H, W)
            if bmode.shape[0] != 1:
                raise ValueError("MEMO expects batch size 1 per adaptation step.")

            img = self._denormalize(bmode[0]).clamp(0.0, 1.0)
            img_pil = transforms.ToPILImage()(img.detach().cpu())

            augmented_bmodes = []
            for _ in range(self.batch_size):
                aug_pil = self.aug_fn(img_pil)
                aug_tensor = transforms.ToTensor()(aug_pil).to(bmode.device)
                augmented_bmodes.append(self._normalize(aug_tensor))

            # Build a batched dict: bmode is (batch_size, C, H, W),
            # other tensors are repeated along batch dim
            aug_x = {}
            for k, v in x.items():
                if k == 'bmode':
                    aug_x[k] = torch.stack(augmented_bmodes, dim=0)
                elif isinstance(v, torch.Tensor):
                    aug_x[k] = v.expand(self.batch_size, *v.shape[1:])
                else:
                    aug_x[k] = v
            return aug_x

        else:
            # Original tensor path (unchanged)
            if x.shape[0] != 1:
                raise ValueError("MEMO expects batch size 1 per adaptation step.")

            img = self._denormalize(x[0]).clamp(0.0, 1.0)
            img_pil = transforms.ToPILImage()(img.detach().cpu())

            augmented = []
            for _ in range(self.batch_size):
                aug_pil = self.aug_fn(img_pil)
                aug_tensor = transforms.ToTensor()(aug_pil).to(x.device)
                augmented.append(self._normalize(aug_tensor))
            return torch.stack(augmented, dim=0)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_memo(
    model: nn.Module,
    lr: float = 2.5e-4,
    steps: int = 1,
    batch_size: int = 32,
    episodic: bool = True,
    architecture: str = "auto",
    layer_selection: str = "auto",
    augmentation_type: str = "augmix",
    image_size: int = 224,
    mean=(0, 0, 0),
    std=(1, 1, 1),
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs: dict | None = None,
) -> Tuple[MEMO, list[str], str]:
    """Configure a model and wrap it with MEMO."""
    optimizer_kwargs = optimizer_kwargs or {"momentum": 0.9}

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
    memo = MEMO(
        model=model,
        optimizer=optimizer,
        steps=steps,
        batch_size=batch_size,
        episodic=episodic,
        augmentation_type=augmentation_type,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    return memo, selected_names, arch