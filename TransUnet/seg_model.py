import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class FrozenSegmentationModel(nn.Module):
    """
    Wrapper for pre-trained MicroSegNet segmentation model.
    Handles batch processing and returns segmentation masks.
    """
    
    def __init__(self, checkpoint_path='MicroSegNet.pth', img_size=224, 
                 n_skip=3, vit_name='R50-ViT-B_16', vit_patches_size=16, 
                 num_classes=1, device='cuda'):
        super().__init__()
        
        self.img_size = img_size
        self.device = device
        
        # Initialize model
        from projects.seg_ttt.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from projects.seg_ttt.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
        
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = n_skip
        config_vit.patches.size = (vit_patches_size, vit_patches_size)
        
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), 
                int(img_size / vit_patches_size)
            )
        
        self.net = ViT_seg(
            config_vit, 
            img_size=img_size, 
            num_classes=config_vit.n_classes
        ).to(device)
        
        # Load pre-trained weights
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.net.eval()
        
        # Freeze all parameters
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        """
        Forward pass for batch of images.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W]
                   - B: batch size
                   - C: channels (1 for grayscale, 3 for RGB)
                   - H, W: height and width
        
        Returns:
            seg_masks: torch.Tensor of shape [B, 1, H, W]
                      Binary segmentation masks (logits, not probabilities)
        """
        B, C, H_orig, W_orig = images.shape
        
        # Normalize to [0, 1] if not already
        if images.max() > 1.0:
            images = images / 255.0
        
        # Convert to grayscale if RGB
        if C == 3:
            # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
            images = (
                0.299 * images[:, 0:1, :, :] + 
                0.587 * images[:, 1:2, :, :] + 
                0.114 * images[:, 2:3, :, :]
            )
        
        # Resize to model's expected input size
        if (H_orig, W_orig) != (self.img_size, self.img_size):
            images_resized = F.interpolate(
                images,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            images_resized = images
        
        # Forward pass through network
        with torch.no_grad():
            outputs, _, _, _ = self.net(images_resized)  # [B, 1, img_size, img_size]
        
        # Resize predictions back to original size
        if (H_orig, W_orig) != (self.img_size, self.img_size):
            seg_masks = F.interpolate(
                outputs,
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )
        else:
            seg_masks = outputs
        
        return seg_masks  # Return logits [B, 1, H, W]
    
    def predict_batch(self, batch_dict):
        """
        Predict segmentation masks for a batch dictionary.
        
        Args:
            batch_dict: Dictionary containing 'bmode' key with images
                       Expected shape: [B, C, H, W] as torch.Tensor
        
        Returns:
            seg_masks: torch.Tensor [B, 1, H, W] - segmentation logits
            seg_probs: torch.Tensor [B, 1, H, W] - segmentation probabilities
            seg_binary: torch.Tensor [B, 1, H, W] - binary masks (0 or 1)
        """
        images = batch_dict['bmode'].to(self.device)
        
        # Get logits
        seg_logits = self.forward(images)
        
        # Convert to probabilities
        seg_probs = torch.sigmoid(seg_logits)
        
        # Convert to binary (threshold at 0.5)
        seg_binary = (seg_probs > 0.5).float()
        
        return seg_logits, seg_probs, seg_binary
    
    def __call__(self, x):
        """
        Allow model(x) syntax.
        
        Args:
            x: Either torch.Tensor [B, C, H, W] or dict with 'bmode' key
        
        Returns:
            seg_masks: torch.Tensor [B, 1, H, W] - segmentation logits
        """
        if isinstance(x, dict):
            return self.predict_batch(x)[0]  # Return logits
        else:
            return self.forward(x)