"""
Common utilities for data extraction
"""

import os
import zipfile
import numpy as np
import spectral as spy
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_envi_hsi(hdr_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load ENVI format HSI cube (.hdr + .dat)
    
    Args:
        hdr_path: Path to .hdr file
        
    Returns:
        cube: HSI data array (H, W, C)
        metadata: Dictionary with wavelengths, resolution, etc.
    """
    try:
        img = spy.open_image(hdr_path)
        cube = img.load()
        
        metadata = {
            'wavelengths': img.metadata.get('wavelength', None),
            'bands': img.nbands,
            'lines': img.nrows,
            'samples': img.ncols,
            'interleave': img.metadata.get('interleave', 'unknown'),
            'data_type': img.dtype
        }
        
        logger.info(f"Loaded ENVI cube: shape={cube.shape}, dtype={cube.dtype}")
        return cube, metadata
        
    except Exception as e:
        logger.error(f"Failed to load ENVI HSI from {hdr_path}: {e}")
        raise


def load_bil_hsi(bil_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load BIL format HSI cube (.bil + .bil.hdr)
    
    Args:
        bil_path: Path to .bil file
        
    Returns:
        cube: HSI data array (H, W, C)
        metadata: Dictionary with metadata
    """
    try:
        hdr_path = str(bil_path) + ".hdr"
        
        if not os.path.exists(hdr_path):
            raise FileNotFoundError(f"Header file not found: {hdr_path}")
        
        img = spy.open_image(hdr_path)
        cube = img.load()
        
        metadata = {
            'wavelengths': img.metadata.get('wavelength', None),
            'bands': img.nbands,
            'lines': img.nrows,
            'samples': img.ncols,
            'interleave': 'bil',
            'data_type': img.dtype
        }
        
        logger.info(f"Loaded BIL cube: shape={cube.shape}, dtype={cube.dtype}")
        return cube, metadata
        
    except Exception as e:
        logger.error(f"Failed to load BIL HSI from {bil_path}: {e}")
        raise


def extract_zip_safe(zip_path: str, output_dir: str, 
                     password: Optional[str] = None) -> bool:
    """
    Safely extract zip file with error handling
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract to
        password: Optional password for encrypted zips
        
    Returns:
        success: True if extraction successful
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if password:
                zip_ref.setpassword(password.encode())
            
            # Get total size for progress
            total_size = sum(f.file_size for f in zip_ref.filelist)
            
            logger.info(f"Extracting {zip_path} ({total_size / 1e9:.2f} GB)")
            zip_ref.extractall(output_dir)
            
        logger.info(f"Successfully extracted to {output_dir}")
        return True
        
    except zipfile.BadZipFile:
        logger.error(f"Corrupted zip file: {zip_path}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed for {zip_path}: {e}")
        return False


def verify_extraction(expected_files: list, base_dir: str) -> bool:
    """
    Verify that all expected files were extracted
    
    Args:
        expected_files: List of expected filenames or patterns
        base_dir: Base directory to check
        
    Returns:
        success: True if all files found
    """
    base_path = Path(base_dir)
    missing_files = []
    
    for pattern in expected_files:
        matches = list(base_path.glob(pattern))
        if not matches:
            missing_files.append(pattern)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    
    logger.info("All expected files verified")
    return True


def get_file_list(directory: str, extension: str) -> list:
    """
    Get list of files with specific extension
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.hdr')
        
    Returns:
        file_list: List of file paths
    """
    path = Path(directory)
    file_list = sorted(path.rglob(f"*{extension}"))
    logger.info(f"Found {len(file_list)} {extension} files in {directory}")
    return file_list


def estimate_storage_size(n_cubes: int, cube_shape: Tuple[int, int, int],
                         dtype: np.dtype = np.float32) -> float:
    """
    Estimate storage size in GB
    
    Args:
        n_cubes: Number of HSI cubes
        cube_shape: Shape (H, W, C) of each cube
        dtype: Data type
        
    Returns:
        size_gb: Estimated size in gigabytes
    """
    bytes_per_element = np.dtype(dtype).itemsize
    elements_per_cube = np.prod(cube_shape)
    total_bytes = n_cubes * elements_per_cube * bytes_per_element
    size_gb = total_bytes / 1e9
    
    logger.info(f"Estimated storage: {size_gb:.2f} GB for {n_cubes} cubes")
    return size_gb
