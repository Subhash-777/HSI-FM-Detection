"""
Loss functions (binary) for logits.

Key fix:
- Support pos_weight for class imbalance (critical for tiny anomalies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    gamma=2.0 focuses on hard examples
    alpha=0.75 upweights positives (anomalies)
    """
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none'
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Focal Loss + Dice Loss combo — best for detection tasks.
    Dice directly optimizes F1-like overlap.
    """
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05,
                 focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, label_smoothing)
        self.focal_w = focal_weight
        self.dice_w = dice_weight

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2 * inter + smooth) / (union + smooth)

    def forward(self, logits, targets):
        return (self.focal_w * self.focal(logits, targets) +
                self.dice_w  * self.dice_loss(logits, targets))

