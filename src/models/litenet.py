"""
LiteNet: Lightweight Dual-Branch Network
"""

import torch
import torch.nn as nn
import logging

from .spectral_branch import SpectralBranch1D
from .spatial_branch import SpatialBranch2D
from .fusion_net import FusionNet

logger = logging.getLogger(__name__)


class LiteNet(nn.Module):
    """Lightweight dual-branch network for HSI anomaly detection"""
    
    def __init__(
        self,
        n_bands: int = 204,
        spatial_channels: int = 3,
        spectral_output_dim: int = 128,
        spatial_output_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
        spectral_architecture: str = "simple",
        spatial_input_size: int = 3
    ):
        super().__init__()
        
        # Spectral Branch
        self.spectral_branch = SpectralBranch1D(
            n_bands=n_bands,
            output_dim=spectral_output_dim,
            architecture=spectral_architecture
        )
        
        # Spatial Branch
        self.spatial_branch = SpatialBranch2D(
            input_channels=spatial_channels,
            output_dim=spatial_output_dim,
            input_size=spatial_input_size,
            mode="patch"
        )
        
        # Fusion Network
        fusion_input_dim = spectral_output_dim + spatial_output_dim
        self.fusion_net = FusionNet(
            input_dim=fusion_input_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=1,
            dropout=dropout
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"SpectralBranch1D: n_bands={n_bands}, output_dim={spectral_output_dim}")
        logger.info(f"SpatialBranch2D: output_dim={spatial_output_dim}")
        logger.info(f"FusionNet: input={fusion_input_dim}, hidden={fusion_hidden_dim}")
        logger.info(f"LiteNet: Total params={total_params:,}")
    
    def forward(self, spectrum, spatial):
        """
        Args:
            spectrum: [B, 204]
            spatial: [B, 3, 3, 3]
        Returns:
            logits: [B, 1]
        """
        spectral_feat = self.spectral_branch(spectrum)
        spatial_feat = self.spatial_branch(spatial)
        logits = self.fusion_net(spectral_feat, spatial_feat)
        return logits
