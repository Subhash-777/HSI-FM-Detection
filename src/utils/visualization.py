"""
Visualization Utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class HSIVisualizer:
    """
    Visualize hyperspectral data
    """
    
    def __init__(self):
        sns.set_style("whitegrid")
    
    def plot_rgb(self, cube: np.ndarray, 
                 band_indices: tuple = None,
                 save_path: Optional[str] = None):
        """
        Plot RGB representation
        
        Args:
            cube: HSI cube (H, W, C)
            band_indices: Indices for R, G, B bands
            save_path: Path to save figure
        """
        if band_indices is None:
            # Default: use bands at 650nm, 550nm, 450nm
            C = cube.shape[2]
            band_indices = (
                int(0.66 * C),  # Red
                int(0.33 * C),  # Green
                int(0.08 * C)   # Blue
            )
        
        rgb = np.stack([
            cube[:, :, band_indices[0]],
            cube[:, :, band_indices[1]],
            cube[:, :, band_indices[2]]
        ], axis=2)
        
        # Normalize
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.title('RGB Representation')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_spectral_bands(self, cube: np.ndarray,
                           n_bands_to_show: int = 9,
                           save_path: Optional[str] = None):
        """
        Plot multiple spectral bands
        
        Args:
            cube: HSI cube
            n_bands_to_show: Number of bands to display
            save_path: Path to save
        """
        H, W, C = cube.shape
        
        # Select evenly spaced bands
        band_indices = np.linspace(0, C-1, n_bands_to_show, dtype=int)
        
        n_rows = int(np.ceil(np.sqrt(n_bands_to_show)))
        n_cols = int(np.ceil(n_bands_to_show / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, band_idx in enumerate(band_indices):
            axes[idx].imshow(cube[:, :, band_idx], cmap='gray')
            axes[idx].set_title(f'Band {band_idx}')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(band_indices), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_mean_spectrum(self, cube: np.ndarray,
                          wavelengths: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
        """
        Plot mean spectrum of entire cube
        
        Args:
            cube: HSI cube
            wavelengths: Wavelength array
            save_path: Path to save
        """
        mean_spectrum = cube.mean(axis=(0, 1))
        std_spectrum = cube.std(axis=(0, 1))
        
        if wavelengths is None:
            wavelengths = np.arange(len(mean_spectrum))
        
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, mean_spectrum, linewidth=2)
        plt.fill_between(
            wavelengths,
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            alpha=0.3
        )
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('Mean Spectral Signature')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def plot_spectral_curve(spectrum: np.ndarray,
                       wavelengths: Optional[np.ndarray] = None,
                       title: str = 'Spectral Curve',
                       save_path: Optional[str] = None):
    """
    Plot a single spectral curve
    
    Args:
        spectrum: Spectral values (C,)
        wavelengths: Wavelength array
        title: Plot title
        save_path: Path to save
    """
    if wavelengths is None:
        wavelengths = np.linspace(400, 1000, len(spectrum))
    
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, spectrum, linewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
