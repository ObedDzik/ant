"""
Shared model definitions for TTA experiments.
"""

import copy

import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import resnet10t


def create_resnet10(in_chans=1, num_classes=2, use_group_norm=True):
    """Create a ResNet10 with GroupNorm (needed for TTA -- BN stats are unreliable)."""
    if use_group_norm:
        model = resnet10t(
            in_chans=in_chans,
            num_classes=num_classes,
            norm_layer=lambda chans: nn.GroupNorm(min(8, chans), chans),
        )
    else:
        model = resnet10t(in_chans=in_chans, num_classes=num_classes)
    return model


class BYOLProjector(nn.Module):
    """MLP projector / predictor for BYOL."""

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TTTModel(nn.Module):
    """
    ResNet10 backbone + classification head + BYOL self-supervised head.

    During source training both heads are trained jointly:
        L = L_CE + alpha * L_BYOL

    At test time (TTT), only the BYOL loss adapts the backbone.
    """

    def __init__(self, in_chans=1, num_classes=2, feature_dim=512, proj_dim=128):
        super().__init__()
        self.backbone = create_resnet10(in_chans=in_chans, num_classes=num_classes)

        self.classifier = copy.deepcopy(self.backbone.fc)
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim

        self.projector = BYOLProjector(feature_dim, 256, proj_dim)
        self.predictor = BYOLProjector(proj_dim, 256, proj_dim)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

    def forward_byol(self, x1, x2):
        """Symmetric BYOL cosine-similarity loss between two views."""
        z1 = self.projector(self.forward_features(x1))
        z2 = self.projector(self.forward_features(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return (
            2
            - F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
            - F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        )


def mutual_information_loss(logits_a, logits_b):
    """
    MI(Y_a, Y_b) = KL( p(y_a, y_b) || p(y_a) p(y_b) )
    Encourages ensemble members to make diverse predictions.
    """
    p_a = F.softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)

    p_joint = (p_a.unsqueeze(2) * p_b.unsqueeze(1)).mean(dim=0)  # (C, C)
    p_a_marg = p_a.mean(dim=0)
    p_b_marg = p_b.mean(dim=0)
    p_prod = p_a_marg.unsqueeze(1) * p_b_marg.unsqueeze(0)

    eps = 1e-8
    return (p_joint * (
        (p_joint + eps).log() - (p_prod + eps).log()
    )).sum()
