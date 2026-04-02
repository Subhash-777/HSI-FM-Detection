"""
Data augmentation for hyperspectral pixel-level training
"""

import torch
import numpy as np
from typing import Dict


class PixelAugmentation:
    """
    Augmentation for preprocessed pixel samples (spectrum + spatial patch).
    """
    
    def __init__(
        self,
        spectral_jitter_std: float = 0.02,
        band_dropout_prob: float = 0.05,
        spatial_rotation: bool = True,
        spatial_flip: bool = True,
        gaussian_noise_std: float = 0.01,
        apply_prob: float = 0.5,
    ):
        self.spectral_jitter_std = spectral_jitter_std
        self.band_dropout_prob = band_dropout_prob
        self.spatial_rotation = spatial_rotation
        self.spatial_flip = spatial_flip
        self.gaussian_noise_std = gaussian_noise_std
        self.apply_prob = apply_prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to a sample.
        
        Args:
            sample: dict with keys 'spectrum', 'spatial', 'label'
                spectrum: (204,) float32
                spatial: (3, 3, 3) float32 [C, H, W]
                label: scalar float32
        
        Returns:
            augmented sample (same structure)
        """
        if np.random.rand() > self.apply_prob:
            return sample
        
        spectrum = sample["spectrum"].clone()
        spatial = sample["spatial"].clone()
        label = sample["label"]
        
        # 1) Spectral jitter (additive Gaussian noise)
        if self.spectral_jitter_std > 0:
            spectrum += torch.randn_like(spectrum) * self.spectral_jitter_std
        
        # 2) Band dropout (randomly zero out bands)
        if self.band_dropout_prob > 0:
            mask = torch.rand(spectrum.shape[0]) > self.band_dropout_prob
            spectrum = spectrum * mask.float()
        
        # 3) Gaussian noise on spatial
        if self.gaussian_noise_std > 0:
            spatial += torch.randn_like(spatial) * self.gaussian_noise_std
        
        # 4) Spatial rotation (90, 180, 270 degrees)
        if self.spatial_rotation and np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
            spatial = torch.rot90(spatial, k=k, dims=(1, 2))
        
        # 5) Spatial flip (horizontal or vertical)
        if self.spatial_flip:
            if np.random.rand() > 0.5:
                spatial = torch.flip(spatial, dims=[1])  # vertical flip
            if np.random.rand() > 0.5:
                spatial = torch.flip(spatial, dims=[2])  # horizontal flip
        
        return {
            "spectrum": spectrum,
            "spatial": spatial,
            "label": label,
        }


class NoAugmentation:
    """Dummy augmentation (returns input unchanged)"""
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return sample
