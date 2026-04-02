"""
Signature Visualization
Visualize and compare FM spectral signatures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SignatureVisualizer:
    """
    Visualize FM spectral signatures
    """
    
    def __init__(self, wavelengths: Optional[np.ndarray] = None):
        """
        Args:
            wavelengths: Wavelength array (nm)
        """
        if wavelengths is None:
            self.wavelengths = np.linspace(400, 1000, 204)
        else:
            self.wavelengths = wavelengths
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_signature(self, signature: Dict, 
                      fm_class: str,
                      save_path: Optional[str] = None):
        """
        Plot a single FM signature with confidence bands
        
        Args:
            signature: Dictionary with mean, std, etc.
            fm_class: FM class name
            save_path: Path to save figure
        """
        fig, ax = plt.subplots()
        
        mean = signature['mean']
        std = signature['std']
        
        # Plot mean spectrum
        ax.plot(self.wavelengths, mean, label='Mean', linewidth=2)
        
        # Plot confidence band (mean ± std)
        ax.fill_between(
            self.wavelengths,
            mean - std,
            mean + std,
            alpha=0.3,
            label='±1 std'
        )
        
        # Plot percentiles
        if 'p25' in signature and 'p75' in signature:
            ax.plot(self.wavelengths, signature['p25'], 
                   '--', alpha=0.5, label='25th percentile')
            ax.plot(self.wavelengths, signature['p75'], 
                   '--', alpha=0.5, label='75th percentile')
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title(f'{fm_class.capitalize()} Spectral Signature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_signatures(self, signatures: Dict,
                                save_path: Optional[str] = None):
        """
        Plot multiple FM signatures for comparison
        
        Args:
            signatures: Dictionary of FM signatures
            save_path: Path to save figure
        """
        fig, ax = plt.subplots()
        
        colors = sns.color_palette("husl", len(signatures))
        
        for (fm_class, signature), color in zip(signatures.items(), colors):
            mean = signature['mean']
            ax.plot(self.wavelengths, mean, label=fm_class.capitalize(), 
                   linewidth=2, color=color)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title('Foreign Material Spectral Signatures')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_signature_heatmap(self, signatures: Dict,
                              save_path: Optional[str] = None):
        """
        Plot heatmap of FM signatures
        
        Args:
            signatures: Dictionary of FM signatures
            save_path: Path to save figure
        """
        # Prepare data matrix
        class_names = list(signatures.keys())
        data_matrix = np.array([signatures[c]['mean'] for c in class_names])
        
        fig, ax = plt.subplots(figsize=(14, len(class_names)))
        
        # Create heatmap
        sns.heatmap(
            data_matrix,
            xticklabels=self.wavelengths[::20].astype(int),
            yticklabels=[c.capitalize() for c in class_names],
            cmap='viridis',
            cbar_kws={'label': 'Reflectance'},
            ax=ax
        )
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title('FM Spectral Signatures Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_separability_matrix(self, signatures: Dict,
                                save_path: Optional[str] = None):
        """
        Plot pairwise separability matrix
        
        Args:
            signatures: Dictionary of FM signatures
            save_path: Path to save figure
        """
        from .spectral_stats import SpectralStats
        
        class_names = list(signatures.keys())
        n_classes = len(class_names)
        
        # Compute pairwise SAM
        sam_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i == j:
                    sam_matrix[i, j] = 0
                else:
                    sam = SpectralStats.compute_spectral_angle(
                        signatures[class1]['mean'],
                        signatures[class2]['mean']
                    )
                    sam_matrix[i, j] = np.degrees(sam)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            sam_matrix,
            xticklabels=[c.capitalize() for c in class_names],
            yticklabels=[c.capitalize() for c in class_names],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Spectral Angle (degrees)'},
            ax=ax
        )
        
        ax.set_title('Pairwise Spectral Angle Mapper (SAM)')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Example usage"""
    from .fm_signature_extractor import FMSignatureExtractor
    
    # Load signatures
    extractor = FMSignatureExtractor()
    signatures = extractor.load_signatures(
        'data/processed/spectral_signatures/fm_signatures.npz'
    )
    
    # Visualize
    visualizer = SignatureVisualizer()
    
    # Plot individual signatures
    for fm_class, signature in signatures.items():
        visualizer.plot_signature(
            signature=signature,
            fm_class=fm_class,
            save_path=f'outputs/figures/signatures/{fm_class}_signature.png'
        )
    
    # Plot all together
    visualizer.plot_multiple_signatures(
        signatures=signatures,
        save_path='outputs/figures/signatures/all_signatures.png'
    )
    
    # Plot heatmap
    visualizer.plot_signature_heatmap(
        signatures=signatures,
        save_path='outputs/figures/signatures/signature_heatmap.png'
    )
    
    # Plot separability
    visualizer.plot_separability_matrix(
        signatures=signatures,
        save_path='outputs/figures/signatures/separability_matrix.png'
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
