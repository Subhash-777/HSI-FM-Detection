"""
Batch Processor
Memory-efficient batch processing of HSI datasets
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List
import yaml
import logging

from .hsi_loader import HSILoader
from .spectral_harmonizer import SpectralHarmonizer
from .spatial_normalizer import SpatialNormalizer
from .radiometric_corrector import RadiometricCorrector

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process HSI datasets in batches with full pipeline
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.loader = HSILoader(cache_size=10)
        
        preproc_config = self.config['preprocessing']
        
        self.harmonizer = SpectralHarmonizer(
            target_bands=preproc_config['target_bands'],
            target_range=preproc_config['spectral_range'],
            method=preproc_config['interpolation']
        )
        
        self.normalizer = SpatialNormalizer(
            target_size=tuple(preproc_config['target_resolution'])
        )
        
        self.corrector = RadiometricCorrector(
            method=preproc_config['normalization']['method']
        )
        
        self.batch_size = preproc_config.get('batch_size', 50)
        
        logger.info("BatchProcessor initialized")
    
    def process_dataset(self, 
                       input_hdf5: str,
                       output_hdf5: str,
                       dataset_name: str,
                       source_bands: int = 204,
                       source_resolution: tuple = (512, 512)):
        """
        Process entire dataset with full pipeline
        
        Args:
            input_hdf5: Input HDF5 file path
            output_hdf5: Output HDF5 file path
            dataset_name: Dataset name ('hsifood', 'agrifood')
            source_bands: Number of bands in source data
            source_resolution: Source spatial resolution
        """
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Input: {input_hdf5}")
        logger.info(f"Output: {output_hdf5}")
        
        # Get dataset info
        info = self.loader.get_hdf5_info(input_hdf5)
        n_samples = info['n_samples']
        
        logger.info(f"Found {n_samples} samples")
        
        # Create output directory
        Path(output_hdf5).parent.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        with h5py.File(output_hdf5, 'w') as hf_out:
            for batch_start in tqdm(range(0, n_samples, self.batch_size),
                                   desc=f"Processing {dataset_name}"):
                batch_end = min(batch_start + self.batch_size, n_samples)
                batch_indices = list(range(batch_start, batch_end))
                
                # Load batch
                batch_samples = self.loader.load_batch_from_hdf5(
                    input_hdf5, batch_indices
                )
                
                # Process each sample
                for idx, (cube, metadata) in enumerate(batch_samples):
                    global_idx = batch_start + idx
                    
                    try:
                        # Process cube
                        processed_cube, processed_metadata = self.process_cube(
                            cube=cube,
                            metadata=metadata,
                            source_bands=source_bands,
                            dataset_name=dataset_name
                        )
                        
                        # Save to output HDF5
                        grp = hf_out.create_group(f"sample_{global_idx:04d}")
                        
                        grp.create_dataset(
                            'hsi_cube',
                            data=processed_cube,
                            compression='gzip',
                            compression_opts=4
                        )
                        
                        # Save metadata
                        for key, value in processed_metadata.items():
                            if isinstance(value, (str, int, float)):
                                grp.attrs[key] = value
                            elif isinstance(value, np.ndarray):
                                if value.dtype == np.bool_:
                                    grp.create_dataset(key, data=value.astype(np.uint8))
                                else:
                                    grp.create_dataset(key, data=value)
                    
                    except Exception as e:
                        logger.warning(f"Failed to process sample {global_idx}: {e}")
                        continue
        
        logger.info(f"Processing complete. Output saved to {output_hdf5}")
    
    def process_cube(self, 
                    cube: np.ndarray,
                    metadata: Dict,
                    source_bands: int,
                    dataset_name: str) -> tuple:
        """
        Process a single HSI cube through full pipeline
        
        Args:
            cube: Input HSI cube
            metadata: Sample metadata
            source_bands: Number of bands in source
            dataset_name: Dataset name
            
        Returns:
            processed_cube: Processed cube
            processed_metadata: Updated metadata
        """
        # 1. Spectral harmonization
        if source_bands != self.config['preprocessing']['target_bands']:
            if dataset_name == 'agrifood':
                cube = self.harmonizer.harmonize_agrifood_300_to_204(cube)
            else:
                cube = self.harmonizer.harmonize_cube(cube)
        
        # 2. Spatial normalization
        cube = self.normalizer.resize_cube(cube)
        
        # 3. Radiometric correction
        cube = self.corrector.normalize_cube(cube, clip_percentile=99.5)
        
        # Update metadata
        processed_metadata = metadata.copy()
        processed_metadata['processed'] = True
        processed_metadata['target_bands'] = self.config['preprocessing']['target_bands']
        processed_metadata['target_resolution'] = self.config['preprocessing']['target_resolution']
        processed_metadata['normalization'] = self.config['preprocessing']['normalization']['method']
        
        # Process mask if present
        if 'mask' in metadata:
            processed_metadata['mask'] = self.normalizer.resize_mask(metadata['mask'])
        
        return cube, processed_metadata
    
    def compute_pca_projection(self, hdf5_path: str, 
                              n_components: int = 3,
                              n_samples_for_fit: int = 100) -> np.ndarray:
        """
        Compute PCA projection for dimensionality reduction
        
        Args:
            hdf5_path: Path to HDF5 file
            n_components: Number of PCA components
            n_samples_for_fit: Number of samples to use for fitting PCA
            
        Returns:
            pca_model: Fitted PCA model
        """
        from sklearn.decomposition import PCA
        
        logger.info(f"Computing PCA with {n_components} components")
        
        # Load subset of samples for PCA fitting
        info = self.loader.get_hdf5_info(hdf5_path)
        n_total = info['n_samples']
        
        sample_indices = np.random.choice(
            n_total, 
            min(n_samples_for_fit, n_total), 
            replace=False
        )
        
        samples = self.loader.load_batch_from_hdf5(hdf5_path, sample_indices.tolist())
        
        # Collect all pixels
        all_pixels = []
        for cube, _ in samples:
            H, W, C = cube.shape
            pixels = cube.reshape(-1, C)
            # Subsample pixels
            subsample_indices = np.random.choice(
                len(pixels), 
                min(1000, len(pixels)), 
                replace=False
            )
            all_pixels.append(pixels[subsample_indices])
        
        all_pixels = np.vstack(all_pixels)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_pixels)
        
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Save PCA model
        import joblib
        pca_path = Path(self.config['paths']['processed_data']) / "pca_models" / "pca_model.pkl"
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, pca_path)
        
        logger.info(f"PCA model saved to {pca_path}")
        
        return pca


def main():
    """Example usage"""
    processor = BatchProcessor('config/config.yaml')
    
    # Process HSIFoodIngr-64
    processor.process_dataset(
        input_hdf5='data/processed/hsifood_clean_samples.h5',
        output_hdf5='data/processed/harmonized_204bands/hsifood_processed.h5',
        dataset_name='hsifood',
        source_bands=204,
        source_resolution=(512, 512)
    )
    
    # Process AgriFoodAnomaly splits
    for split in ['train', 'val', 'test']:
        processor.process_dataset(
            input_hdf5=f'data/processed/agrifood_{split}.h5',
            output_hdf5=f'data/processed/harmonized_204bands/agrifood_{split}_processed.h5',
            dataset_name='agrifood',
            source_bands=300,
            source_resolution=(1000, 900)
        )
    
    # Compute PCA projection
    processor.compute_pca_projection(
        hdf5_path='data/processed/harmonized_204bands/hsifood_processed.h5',
        n_components=3,
        n_samples_for_fit=100
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()