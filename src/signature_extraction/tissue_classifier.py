"""
Tissue Classifier
GMM-based food tissue segmentation (muscle vs fat)
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TissueClassifier:
    """
    Classify food tissues using GMM for spectral clustering
    """
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Args:
            n_components: Number of tissue types (typically 2: muscle, fat)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        
        logger.info(f"TissueClassifier initialized: n_components={n_components}")
    
    def fit_gmm(self, cube: np.ndarray, 
                subsample_ratio: float = 0.1) -> GaussianMixture:
        """
        Fit GMM to cube spectral data
        
        Args:
            cube: HSI cube (H, W, C)
            subsample_ratio: Ratio of pixels to use for fitting
            
        Returns:
            gmm: Fitted GMM model
        """
        H, W, C = cube.shape
        
        # Reshape to (n_pixels, n_bands)
        pixels = cube.reshape(-1, C)
        
        # Subsample for efficiency
        n_samples = int(len(pixels) * subsample_ratio)
        indices = np.random.choice(len(pixels), n_samples, replace=False)
        sampled_pixels = pixels[indices]
        
        logger.info(f"Fitting GMM on {n_samples} pixels...")
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        )
        self.gmm.fit(sampled_pixels)
        
        logger.info("GMM fitting complete")
        
        return self.gmm
    
    def classify_cube(self, cube: np.ndarray) -> np.ndarray:
        """
        Classify cube pixels into tissue types
        
        Args:
            cube: HSI cube (H, W, C)
            
        Returns:
            labels: Tissue labels (H, W)
        """
        if self.gmm is None:
            logger.warning("GMM not fitted. Fitting now...")
            self.fit_gmm(cube)
        
        H, W, C = cube.shape
        pixels = cube.reshape(-1, C)
        
        # Predict labels
        labels = self.gmm.predict(pixels)
        labels = labels.reshape(H, W)
        
        return labels
    
    def get_tissue_probabilities(self, cube: np.ndarray) -> np.ndarray:
        """
        Get tissue membership probabilities
        
        Args:
            cube: HSI cube (H, W, C)
            
        Returns:
            probabilities: Tissue probabilities (H, W, n_components)
        """
        if self.gmm is None:
            raise ValueError("GMM not fitted")
        
        H, W, C = cube.shape
        pixels = cube.reshape(-1, C)
        
        # Predict probabilities
        probs = self.gmm.predict_proba(pixels)
        probs = probs.reshape(H, W, self.n_components)
        
        return probs
    
    def identify_tissue_boundaries(self, labels: np.ndarray, 
                                   kernel_size: int = 3) -> np.ndarray:
        """
        Identify boundaries between tissue types
        
        Args:
            labels: Tissue labels (H, W)
            kernel_size: Size of kernel for boundary detection
            
        Returns:
            boundaries: Binary mask of boundaries (H, W)
        """
        import cv2
        
        # Dilate and erode to find boundaries
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        dilated = cv2.dilate(labels.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(labels.astype(np.uint8), kernel, iterations=1)
        
        boundaries = (dilated != eroded).astype(np.uint8)
        
        return boundaries


def main():
    """Example usage"""
    # Load a sample cube
    import h5py
    
    with h5py.File('data/processed/harmonized_204bands/hsifood_processed.h5', 'r') as hf:
        cube = hf['sample_0000/hsi_cube'][:]
    
    # Classify tissues
    classifier = TissueClassifier(n_components=2)
    labels = classifier.classify_cube(cube)
    
    # Get boundaries
    boundaries = classifier.identify_tissue_boundaries(labels)
    
    print(f"Tissue labels shape: {labels.shape}")
    print(f"Unique tissue types: {np.unique(labels)}")
    print(f"Boundary pixels: {boundaries.sum()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
