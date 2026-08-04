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

class LightSegHead(nn.Module):
    def __init__(self, encoder_dim=1024, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(encoder_dim, hidden_dim, kernel_size=3, padding=1), 
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
    def forward(self, features):
        return self.head(features)

class LinearProbeSegHead(nn.Module):
    def __init__(self, encoder_dim=1024):
        super().__init__()
        self.head = nn.Conv2d(encoder_dim, 1, kernel_size=1)
    def forward(self, x):
        return self.head(x)



class SupervisedTrainerWithSegTTT:
    """
    Standard supervised training for cancer detection.
    Test-time adaptation using frozen segmentation model.
    """
    
    def __init__(self, model, frozen_segnet, cfg):
        """
        Args:
            model: Your cancer detection model (DINOv3 + UNETR + ProstNFoundMeta)
            frozen_segnet: Frozen pre-trained segmentation model (e.g., MicroSegNet)
            cfg: Configuration object
        """
        self.model = model
        self.use_dino = cfg.get('use_dino', True)
        if self.use_dino:
            self.image_encoder = model.model.model.image_encoder.backbone
        else:
            self.image_encoder = model.model.medsam_model.image_encoder
        self.frozen_segnet = frozen_segnet
        self.frozen_segnet.eval()  # Keep frozen
        for param in self.frozen_segnet.parameters():
            param.requires_grad = False
        
        self.cfg = cfg
        self.device = cfg.device
        self.dice_metric = DiceMetric(include_background=True, reduction='mean')
        self.ttt_seg_loss = DiceCELoss(sigmoid=True, reduction='mean')
        self._ttt_optimizer = None
        self._all_params = None
 
        self.inner_lr = cfg.get('inner_lr', 1e-4)
        self.inner_steps = cfg.get('inner_steps', 10)
        self.layer_selection = cfg.get('layer_selection', 'auto')
        self.n_blocks = cfg.get('n_blocks', 4)
        seghead = cfg.get('seghead', None)
        
        encoder_dim = cfg.get('encoder_dim', 1024)

        if seghead == 'linear_probe':
            self.ttt_seg_head = LinearProbeSegHead(encoder_dim=encoder_dim).to(self.device)

        elif seghead == 'light':
            self.ttt_seg_head = LightSegHead(encoder_dim=encoder_dim, hidden_dim=256).to(self.device)

        else:
            self.ttt_seg_head = SegHead(encoder_dim=encoder_dim, hidden_dim=256).to(self.device)


    def reconfigure_model(self, model):
        """Re-run model configuration after training loop restores model state.
        Called at the start of each val epoch to re-null BN buffers and 
        re-set requires_grad, without touching the optimizer or seg head.
        """
        model, selected_names, arch = configure_model_for_adaptation(
            model=model,
            architecture='auto',
            layer_selection=self.layer_selection,
        )
        check_adaptation_ready(model)
        return model

    def get_or_create_optimizer(self, model):
        """Create optimizer once — reused across all batches and epochs."""
        if self._ttt_optimizer is not None:
            return model, self._ttt_optimizer, self._all_params

        # First call only — create optimizer
        params, _, _ = collect_adaptation_params(
            model=model,
            architecture='auto',
            layer_selection=self.layer_selection,
        )
        seg_head_params = list(self.ttt_seg_head.parameters())
        all_params = params + seg_head_params

        self._ttt_optimizer = torch.optim.Adam([
            {'params': params, 'lr': self.inner_lr * 0.1},
            {'params': seg_head_params, 'lr': self.inner_lr},
        ])
        self._all_params = all_params
        return model, self._ttt_optimizer, all_params
    
    def get_encoder_features(self, model, image):
        """
        Extract encoder features for segmentation head.
        Args:
            model: Cancer detection model
            image: Input image [B, 3, H, W]
        Returns:
            features: [B, C, H_feat, W_feat] encoder features
        """
        with torch.set_grad_enabled(True):
            features = self.image_encoder.get_intermediate_layers(image, n=1)[0]  # [B, N, C]
            B, N, C = features.shape
            # Reshape to spatial: [B, C, H, W]
            H_img, W_img = image.shape[2], image.shape[3]
            patch_size = 16
            H_feat = H_img // patch_size
            W_feat = W_img // patch_size
            features = features.permute(0, 2, 1)  # [B, C, N]
            features = features.reshape(B, C, H_feat, W_feat)  # [B, C, H, W]
        
        return features
    
    def pnf_get_encoder_features(self, model, image):
        """Extract features by hooking into normal forward pass."""
        features_out = {}

        def hook_fn(module, input, output):
            features_out['features'] = output

        # Register hook on last transformer block
        # image_encoder = model.model.medsam_model.image_encoder
        hook = self.image_encoder.blocks[-1].register_forward_hook(hook_fn)

        with torch.set_grad_enabled(True):
            # Run normal encoder forward — handles pos_embed internally
            _ = self.image_encoder(image)

        hook.remove()
        # Output of last block is [B, H, W, C] — convert to [B, C, H, W]
        features = features_out['features'].permute(0, 3, 1, 2)
        return features

    def apply_segmentation_ttt(self, model, batch, val_iter):
        """Apply test-time training with LoRA or direct fine-tuning."""
        
        model.train()
        self.ttt_seg_head.train()
        
        #lr = self.inner_lr #if not self.use_lora else self.inner_lr * 10
        model, ttt_optimizer, all_params = self.get_or_create_optimizer(model)

        image = batch['bmode'].to(self.device)
        
        with torch.no_grad():
            pseudo_mask = self.frozen_segnet(image)
            if isinstance(pseudo_mask, dict):
                pseudo_mask = pseudo_mask
            pseudo_mask = torch.sigmoid(pseudo_mask)
        
        seg_losses = []
        dice_scores = []
        
        for step in range(self.inner_steps):
            ttt_optimizer.zero_grad()

            if not self.use_dino:
                encoder_features = self.pnf_get_encoder_features(model, image)
            else:
                encoder_features = self.get_encoder_features(model, image)
            seg_pred = self.ttt_seg_head(encoder_features)
            seg_pred = F.interpolate(seg_pred, size=pseudo_mask.shape[2:], mode='bilinear', align_corners=False)
            
            seg_loss = self.ttt_seg_loss(seg_pred, pseudo_mask)
            seg_losses.append(seg_loss.item())

            with torch.no_grad():
                seg_pred_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                pseudo_mask_binary = (pseudo_mask > 0.5).float()
                self.dice_metric(seg_pred_binary, pseudo_mask_binary)
                dice_score = self.dice_metric.aggregate().item()
                dice_scores.append(dice_score)
                self.dice_metric.reset()

            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            ttt_optimizer.step()

        avg_loss = np.mean(seg_losses) if seg_losses else 0.0
        
        wandb.log({
            "ttt/iter_dice_mean": np.mean(dice_scores),
            "ttt/inner_dice_delta": dice_scores[-1] - dice_scores[0],
            "ttt/mean_loss": avg_loss,
            #"ttt/using_lora": self.use_lora,
        })
        
        return model, dice_scores