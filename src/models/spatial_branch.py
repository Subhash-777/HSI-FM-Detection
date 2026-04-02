"""
Spatial Branch: 2D CNN for spatial context processing
"""

import torch
import torch.nn as nn


class SpatialBranch2D(nn.Module):
    """2D CNN for spatial feature extraction from 3x3 PCA patches"""
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 64,
        input_size: int = 3,
        mode: str = "patch"
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # 2D CNN
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] e.g., [B, 3, 3, 3]
        Returns:
            features: [B, output_dim]
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        x = self.pool(x)  # [B, output_dim, 1, 1]
        x = x.flatten(1)  # [B, output_dim]
        
        return x
