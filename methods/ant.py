import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import logging
import os
from monai.losses.dice import DiceLoss
from monai.losses.dice import DiceCELoss
from monai.metrics import DiceMetric
import wandb


class SegHead(nn.Module):
    def __init__(self, encoder_dim=1024, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(encoder_dim, hidden_dim, kernel_size=3, padding=1), 
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
    def forward(self, features):
        return self.head(features)


class SupervisedTrainerWithSegTTT:
    """
    Test-time adaptation with:
    - Stateless encoder adaptation (per batch)
    - Stateful segmentation head learning (across validation)
    """

    def __init__(self, frozen_segnet, cfg):
        self.device = cfg.device
        self.cfg = cfg
        self.use_dino = cfg.get('use_dino', True)
        self.frozen_segnet = frozen_segnet
        self.frozen_segnet.eval()

        for p in self.frozen_segnet.parameters():
            p.requires_grad = False

        self.dice_metric = DiceMetric(include_background=True, reduction='mean')
        self.ttt_seg_loss = DiceCELoss(sigmoid=True, reduction='mean')
        self.inner_lr = cfg.get('inner_lr', 1e-4)
        self.inner_steps = cfg.get('inner_steps', 10)

        self.update_layers = cfg.get('update_layers', 'first_n')
        self.update_norm = cfg.get('update_norm', False)
        self.update_bias = cfg.get('update_bias', False)
        self.n_blocks = cfg.get('n_blocks', 6)

        # Within val loop persistence flags
        self.persist_encoder = cfg.get('persist_encoder', False)
        self.persist_head = cfg.get('persist_head', True)

        # Across val loop persistence flags (stateful throughout training)
        self.stateful_head = cfg.get('stateful_head', False)
        self.stateful_encoder_optim = cfg.get('stateful_encoder_optim', False)

        self._encoder_optimizer = None
        self._encoder_params = None

        encoder_dim = cfg.get('encoder_dim', 1024)
        self.ttt_seg_head = SegHead(encoder_dim, 256).to(self.device)
        self.seghead_optimizer = torch.optim.Adam(self.ttt_seg_head.parameters(), lr=self.inner_lr)
        
    def _build_seghead(self):
        encoder_dim = self.ttt_seg_head.encoder_dim if hasattr(self.ttt_seg_head, "encoder_dim") else 1024
        return SegHead(encoder_dim, 256).to(self.device)
        
    def reset_objects(self):
        # Always clear param references — refreshed each batch anyway
        self._encoder_params = None
        # Reset encoder optimizer unless it is stateful across val loops
        if not self.stateful_encoder_optim:
            self._encoder_optimizer = None
        # Reset head and its optimizer unless stateful across val loops
        if not self.stateful_head:
            if hasattr(self, 'ttt_seg_head'):
                self.ttt_seg_head = self._build_seghead()
                self.seghead_optimizer = torch.optim.Adam(self.ttt_seg_head.parameters(), lr=self.inner_lr)

    def head_reset_each_batch(self):
        if hasattr(self, 'ttt_seg_head'):
            self.ttt_seg_head = self._build_seghead()
            self.seghead_optimizer = torch.optim.Adam(self.ttt_seg_head.parameters(), lr=self.inner_lr)

    def _get_block_groups(self, blocks):
        n = len(blocks)
        third = n // 3
        return {
            "first": blocks[:third],
            "middle": blocks[third:2*third],
            "last": blocks[2*third:]
        }

    def get_adaptation_params(self, model):
        """
        Returns parameters based on:
        - layer selection (first_n / middle_n / last_n / all)
        - update_norm
        - update_bias
        """
        if not self.use_dino:
            image_encoder = model.model.medsam_model.image_encoder
        else:
            image_encoder = model.model.model.image_encoder.backbone
        if not hasattr(image_encoder, 'blocks'):
            print("Model has no blocks!!!")
            return image_encoder.parameters()

        blocks = image_encoder.blocks
        groups = self._get_block_groups(blocks)
        if self.update_layers == 'all':
            selected_blocks = blocks
        elif self.update_layers == 'first_n':
            selected_blocks = groups["first"][:self.n_blocks]
        elif self.update_layers == 'middle_n':
            selected_blocks = groups["middle"][:self.n_blocks]
        elif self.update_layers == 'last_n':
            selected_blocks = groups["last"][-self.n_blocks:]
        elif self.update_layers == 'last_only':
            selected_blocks = [blocks[-1]]
        else:
            selected_blocks = blocks

        params = []
        for block in selected_blocks:
            for module in block.modules():
                # -------- NORM --------
                if self.update_norm:
                    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                        for p in module.parameters(recurse=False):
                            if p.requires_grad:
                                params.append(p)
                # -------- BIAS --------
                if self.update_bias:
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        if module.bias is not None:
                            params.append(module.bias)
                # -------- FULL --------
                if not self.update_norm and not self.update_bias:
                    # default: full parameter update for selected blocks
                    for p in module.parameters(recurse=False):
                        if p.requires_grad:
                            params.append(p)
        return params

    def get_encoder_features(self, model, image):
        image_encoder = model.model.model.image_encoder.backbone
        features = image_encoder.get_intermediate_layers(image, n=1)[0]
        B, N, C = features.shape
        H = image.shape[2] // 16
        W = image.shape[3] // 16
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        return features
    
    def pnf_get_encoder_features(self, model, image):
        """Extract features by hooking into normal forward pass."""
        features_out = {}
        def hook_fn(module, input, output):
            features_out['features'] = output
        image_encoder = model.model.medsam_model.image_encoder
        hook = image_encoder.blocks[-1].register_forward_hook(hook_fn)
        with torch.set_grad_enabled(True):
            _ = image_encoder(image)
        hook.remove()
        features = features_out['features'].permute(0, 3, 1, 2)
        return features

    def freeze_bn(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def apply_segmentation_ttt(self, model, batch, val_iter):
        model.train()
        self.ttt_seg_head.train()
        # Always refresh parameter references since encoder may have been restored
        self._encoder_params = self.get_adaptation_params(model)
        if self._encoder_optimizer is None or not self.stateful_encoder_optim:
            # Create fresh optimizer with no accumulated state
            self._encoder_optimizer = torch.optim.Adam(self._encoder_params, lr=self.inner_lr * 0.1)
        else:
            # Preserve Adam state but update parameter references
            for group, new_params in zip(
                self._encoder_optimizer.param_groups,
                [self._encoder_params]
            ):
                group['params'] = new_params

        encoder_optimizer = self._encoder_optimizer
        encoder_params = self._encoder_params
        # Reset head each batch only if not persisting within val loop
        if not self.persist_head:
            self.head_reset_each_batch()

        image = batch['bmode'].to(self.device)
        with torch.no_grad():
            pseudo_mask = self.frozen_segnet(image)
            pseudo_mask = torch.sigmoid(pseudo_mask)

        seg_losses = []
        dice_scores = []
        for step in range(self.inner_steps):
            encoder_optimizer.zero_grad()
            self.seghead_optimizer.zero_grad()
            if self.use_dino:
                feats = self.get_encoder_features(model, image)
            else:
                feats = self.pnf_get_encoder_features(model, image)
            seg_pred = self.ttt_seg_head(feats)
            seg_pred = F.interpolate(
                seg_pred,
                size=pseudo_mask.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            loss = self.ttt_seg_loss(seg_pred, pseudo_mask)
            seg_losses.append(loss.item())
            with torch.no_grad():
                pred_bin = (torch.sigmoid(seg_pred) > 0.5).float()
                mask_bin = (pseudo_mask > 0.5).float()
                self.dice_metric(pred_bin, mask_bin)
                dice = self.dice_metric.aggregate().item()
                dice_scores.append(dice)
                self.dice_metric.reset()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            torch.nn.utils.clip_grad_norm_(self.ttt_seg_head.parameters(), 1.0)
            encoder_optimizer.step()
            self.seghead_optimizer.step()

        wandb.log({
            "ttt/iter_dice_mean": np.mean(dice_scores),
            "ttt/inner_dice_delta": dice_scores[-1] - dice_scores[0],
            "ttt/mean_loss": np.mean(seg_losses),
        })
        return dice_scores