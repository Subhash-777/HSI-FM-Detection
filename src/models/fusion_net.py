"""
Fusion Network: Combines spectral and spatial features
"""

import torch
import torch.nn as nn


class FusionNet(nn.Module):
    """Fusion network for combining features"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, spectral_feat, spatial_feat):
        """
        Args:
            spectral_feat: [B, spectral_dim]
            spatial_feat: [B, spatial_dim]
        Returns:
            logits: [B, 1]
        """
        x = torch.cat([spectral_feat, spatial_feat], dim=1)
        
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        
        logits = self.fc_out(x)
        
        return logits
