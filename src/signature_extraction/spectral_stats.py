"""
Spectral Statistics
Compute and analyze spectral statistics
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SpectralStats:
    """
    Compute spectral statistics and comparisons
    """
    
    @staticmethod
    def compute_spectral_angle(spectrum1: np.ndarray, 
                              spectrum2: np.ndarray) -> float:
        """
        Compute Spectral Angle Mapper (SAM) between two spectra
        
        Args:
            spectrum1: First spectrum (C,)
            spectrum2: Second spectrum (C,)
            
        Returns:
            angle: Spectral angle in radians
        """
        dot_product = np.dot(spectrum1, spectrum2)
        norm_product = np.linalg.norm(spectrum1) * np.linalg.norm(spectrum2)
        
        if norm_product == 0:
            return 0.0
        
        cos_angle = dot_product / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        
        return angle
    
    @staticmethod
    def compute_spectral_distance(spectrum1: np.ndarray,
                                 spectrum2: np.ndarray,
                                 metric: str = 'euclidean') -> float:
        """
        Compute distance between two spectra
        
        Args:
            spectrum1: First spectrum
            spectrum2: Second spectrum
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            
        Returns:
            distance: Spectral distance
        """
        if metric == 'euclidean':
            return np.linalg.norm(spectrum1 - spectrum2)
        elif metric == 'manhattan':
            return np.sum(np.abs(spectrum1 - spectrum2))
        elif metric == 'cosine':
            return 1 - np.dot(spectrum1, spectrum2) / (
                np.linalg.norm(spectrum1) * np.linalg.norm(spectrum2)
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def compute_separability(class1_spectra: np.ndarray,
                           class2_spectra: np.ndarray) -> Dict:
        """
        Compute separability metrics between two classes
        
        Args:
            class1_spectra: Spectra for class 1 (N1, C)
            class2_spectra: Spectra for class 2 (N2, C)
            
        Returns:
            metrics: Dictionary with separability metrics
        """
        # Compute means
        mean1 = np.mean(class1_spectra, axis=0)
        mean2 = np.mean(class2_spectra, axis=0)
        
        # Compute covariances
        cov1 = np.cov(class1_spectra.T)
        cov2 = np.cov(class2_spectra.T)
        
        # Spectral angle between means
        sam = SpectralStats.compute_spectral_angle(mean1, mean2)
        
        # Euclidean distance between means
        euclidean = np.linalg.norm(mean1 - mean2)
        
        # Jeffries-Matusita distance (simplified)
        # JM = sqrt(2 * (1 - exp(-B)))
        # where B is the Bhattacharyya distance
        diff = mean1 - mean2
        pooled_cov = (cov1 + cov2) / 2
        
        try:
            inv_pooled_cov = np.linalg.inv(pooled_cov)
            bhattacharyya = 0.125 * diff.T @ inv_pooled_cov @ diff
            jm_distance = np.sqrt(2 * (1 - np.exp(-bhattacharyya)))
        except:
            jm_distance = None
        
        metrics = {
            'spectral_angle_mapper': float(sam),
            'euclidean_distance': float(euclidean),
            'jeffries_matusita_distance': float(jm_distance) if jm_distance else None
        }
        
        return metrics
    
    @staticmethod
    def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio
        
        Args:
            signal: Signal array
            noise: Noise array
            
        Returns:
            snr: SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        return snr


def main():
    """Example usage"""
    # Load FM signatures
    signatures = np.load('data/processed/spectral_signatures/fm_signatures.npz')
    
    # Compare plastic and glass
    plastic_mean = signatures['plastic_mean']
    glass_mean = signatures['glass_mean']
    
    # Compute spectral angle
    sam = SpectralStats.compute_spectral_angle(plastic_mean, glass_mean)
    print(f"SAM between plastic and glass: {sam:.4f} radians ({np.degrees(sam):.2f} degrees)")
    
    # Compute distance
    distance = SpectralStats.compute_spectral_distance(plastic_mean, glass_mean)
    print(f"Euclidean distance: {distance:.4f}")


if __name__ == "__main__":
    main()
