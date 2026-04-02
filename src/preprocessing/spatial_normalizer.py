"""
Spatial Normalization
Resize HSI cubes to target resolution (256x256)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SpatialNormalizer:
    """
    Normalize spatial dimensions of HSI cubes
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            target_size: Target spatial resolution (height, width)
        """
        self.target_size = target_size
        logger.info(f"SpatialNormalizer initialized: target_size={target_size}")
    
    def resize_cube(self, cube: np.ndarray, 
                    interpolation: str = 'linear') -> np.ndarray:
        """
        Resize HSI cube to target spatial resolution
        
        Args:
            cube: Input HSI cube (H, W, C)
            interpolation: Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns:
            resized_cube: Output HSI cube (target_H, target_W, C)
        """
        H, W, C = cube.shape
        target_H, target_W = self.target_size
        
        # Check if already at target size
        if (H, W) == (target_H, target_W):
            logger.debug("Cube already at target size")
            return cube
        
        logger.info(f"Resizing cube: ({H}, {W}) → ({target_H}, {target_W})")
        
        # Map interpolation method
        interp_map = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA
        }
        
        cv_interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        
        # Resize band by band to manage memory
        resized_cube = np.zeros((target_H, target_W, C), dtype=cube.dtype)
        
        for c in range(C):
            resized_cube[:, :, c] = cv2.resize(
                cube[:, :, c],
                (target_W, target_H),
                interpolation=cv_interp
            )
        
        return resized_cube
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize binary mask to target size
        
        Args:
            mask: Input binary mask (H, W)
            
        Returns:
            resized_mask: Output binary mask (target_H, target_W)
        """
        target_H, target_W = self.target_size
        
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (target_W, target_H),
            interpolation=cv2.INTER_NEAREST  # Use nearest for binary masks
        )
        
        return resized_mask.astype(bool)
    
    def resize_batch(self, cubes: list, 
                    interpolation: str = 'linear') -> list:
        """
        Resize a batch of cubes
        
        Args:
            cubes: List of HSI cubes
            interpolation: Interpolation method
            
        Returns:
            resized_cubes: List of resized cubes
        """
        from tqdm import tqdm
        
        resized = []
        for cube in tqdm(cubes, desc="Resizing batch"):
            resized.append(self.resize_cube(cube, interpolation))
        
        return resized
    
    def crop_center(self, cube: np.ndarray, 
                    crop_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Crop center region of cube
        
        Args:
            cube: Input HSI cube (H, W, C)
            crop_size: Size to crop (height, width), defaults to target_size
            
        Returns:
            cropped_cube: Cropped cube
        """
        if crop_size is None:
            crop_size = self.target_size
        
        H, W, C = cube.shape
        crop_H, crop_W = crop_size
        
        if H < crop_H or W < crop_W:
            raise ValueError(f"Cube size ({H}, {W}) smaller than crop size ({crop_H}, {crop_W})")
        
        start_h = (H - crop_H) // 2
        start_w = (W - crop_W) // 2
        
        cropped = cube[start_h:start_h+crop_H, start_w:start_w+crop_W, :]
        
        return cropped
