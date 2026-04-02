"""
Synthetic FM Generator
Main generator for creating synthetic FM-inserted cubes
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
import json

from .insertion_strategies import GaussianInsertion, EdgeBlendingInsertion, RealisticInsertion
from ..signature_extraction.tissue_classifier import TissueClassifier

logger = logging.getLogger(__name__)


class SyntheticFMGenerator:
    """
    Generate synthetic FM-inserted HSI cubes
    """
    
    def __init__(self, 
                 fm_signatures: Dict,
                 insertion_strategy: str = 'realistic',
                 size_range: Tuple[int, int] = (2, 5),
                 n_objects_range: Tuple[int, int] = (3, 7),
                 gmm_aware: bool = True,
                 avoid_edges: bool = True):
        """
        Args:
            fm_signatures: Dictionary of FM spectral signatures
            insertion_strategy: Strategy name ('gaussian', 'edge_blending', 'realistic')
            size_range: Range of FM object sizes (pixels)
            n_objects_range: Range of number of objects per cube
            gmm_aware: Whether to avoid tissue boundaries
            avoid_edges: Whether to avoid image edges
        """
        self.fm_signatures = fm_signatures
        self.fm_classes = list(fm_signatures.keys())
        self.size_range = size_range
        self.n_objects_range = n_objects_range
        self.gmm_aware = gmm_aware
        self.avoid_edges = avoid_edges
        
        # Initialize insertion strategy
        if insertion_strategy == 'gaussian':
            self.inserter = GaussianInsertion()
        elif insertion_strategy == 'edge_blending':
            self.inserter = EdgeBlendingInsertion()
        elif insertion_strategy == 'realistic':
            self.inserter = RealisticInsertion()
        else:
            raise ValueError(f"Unknown strategy: {insertion_strategy}")
        
        # Initialize tissue classifier
        if self.gmm_aware:
            self.tissue_classifier = TissueClassifier(n_components=2)
        
        logger.info(f"SyntheticFMGenerator initialized: strategy={insertion_strategy}, "
                   f"size_range={size_range}, n_objects_range={n_objects_range}")
    
    def generate_single(self, clean_cube: np.ndarray,
                       fm_class: Optional[str] = None,
                       n_objects: Optional[int] = None,
                       size: Optional[int] = None) -> Dict:
        """
        Generate a single synthetic sample
        
        Args:
            clean_cube: Clean HSI cube (H, W, C)
            fm_class: FM class (random if None)
            n_objects: Number of objects (random if None)
            size: Object size (random if None)
            
        Returns:
            sample: Dictionary with synthetic cube, mask, and metadata
        """
        H, W, C = clean_cube.shape
        
        # Random selections
        if fm_class is None:
            fm_class = np.random.choice(self.fm_classes)
        
        if n_objects is None:
            n_objects = np.random.randint(*self.n_objects_range)
        
        # Get tissue boundaries if GMM-aware
        if self.gmm_aware:
            tissue_labels = self.tissue_classifier.classify_cube(clean_cube)
            boundaries = self.tissue_classifier.identify_tissue_boundaries(tissue_labels)
        else:
            boundaries = None
        
        # Initialize
        modified_cube = clean_cube.copy()
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Sample FM spectrum from distribution
        fm_sig = self.fm_signatures[fm_class]
        
        inserted_objects = []
        
        for obj_idx in range(n_objects):
            # Random size if not specified
            if size is None:
                obj_size = np.random.randint(*self.size_range)
            else:
                obj_size = size
            
            # Find valid position
            valid_pos = False
            max_attempts = 50
            
            for attempt in range(max_attempts):
                # Random center position
                if self.avoid_edges:
                    cx = np.random.randint(obj_size + 5, H - obj_size - 5)
                    cy = np.random.randint(obj_size + 5, W - obj_size - 5)
                else:
                    cx = np.random.randint(obj_size, H - obj_size)
                    cy = np.random.randint(obj_size, W - obj_size)
                
                # Check if on tissue boundary
                if boundaries is not None:
                    # Check region around center
                    region = boundaries[
                        max(0, cx-obj_size):min(H, cx+obj_size+1),
                        max(0, cy-obj_size):min(W, cy+obj_size+1)
                    ]
                    if region.sum() > region.size * 0.3:  # More than 30% on boundary
                        continue
                
                # Check if overlaps with existing objects
                test_mask = combined_mask[
                    max(0, cx-obj_size):min(H, cx+obj_size+1),
                    max(0, cy-obj_size):min(W, cy+obj_size+1)
                ]
                if test_mask.sum() == 0:  # No overlap
                    valid_pos = True
                    break
            
            if not valid_pos:
                logger.debug(f"Could not find valid position for object {obj_idx}")
                continue
            
            # Sample spectrum with variance
            fm_spectrum = np.random.normal(
                fm_sig['mean'],
                fm_sig['std'] * 0.3  # 30% of std for variation
            )
            fm_spectrum = np.clip(fm_spectrum, 0, 1)
            
            # Insert FM
            modified_cube, obj_mask = self.inserter.insert(
                cube=modified_cube,
                fm_spectrum=fm_spectrum,
                center=(cx, cy),
                size=obj_size
            )
            
            # Update combined mask
            combined_mask = np.logical_or(combined_mask, obj_mask).astype(np.uint8)
            
            inserted_objects.append({
                'center': (int(cx), int(cy)),
                'size': int(obj_size),
                'fm_class': fm_class
            })
        
        return {
            'cube': modified_cube,
            'mask': combined_mask,
            'fm_class': fm_class,
            'n_objects': len(inserted_objects),
            'objects': inserted_objects
        }
    
    def generate_dataset(self,
                        clean_cubes_hdf5: str,
                        output_hdf5: str,
                        n_synthetic_per_clean: int = 10,
                        max_clean_samples: Optional[int] = None):
        """
        Generate synthetic dataset from clean cubes
        
        Args:
            clean_cubes_hdf5: Path to clean cubes HDF5
            output_hdf5: Output path for synthetic dataset
            n_synthetic_per_clean: Number of synthetic samples per clean cube
            max_clean_samples: Maximum number of clean cubes to use
        """
        logger.info(f"Generating synthetic dataset...")
        logger.info(f"Input: {clean_cubes_hdf5}")
        logger.info(f"Output: {output_hdf5}")
        logger.info(f"Synthetic per clean: {n_synthetic_per_clean}")
        
        # Load clean cubes info
        from ..preprocessing.hsi_loader import HSILoader
        loader = HSILoader()
        info = loader.get_hdf5_info(clean_cubes_hdf5)
        n_clean = info['n_samples']
        
        if max_clean_samples:
            n_clean = min(n_clean, max_clean_samples)
        
        logger.info(f"Using {n_clean} clean cubes")
        
        # Create output directory
        Path(output_hdf5).parent.mkdir(parents=True, exist_ok=True)
        
        # Generation statistics
        stats = {fm_class: 0 for fm_class in self.fm_classes}
        
        with h5py.File(output_hdf5, 'w') as hf_out:
            sample_idx = 0
            
            for clean_idx in tqdm(range(n_clean), desc="Generating synthetic"):
                # Load clean cube
                clean_cube, metadata = loader.load_from_hdf5(
                    clean_cubes_hdf5, clean_idx
                )
                
                # Generate multiple synthetic samples
                for syn_idx in range(n_synthetic_per_clean):
                    try:
                        # Cycle through FM classes
                        fm_class = self.fm_classes[syn_idx % len(self.fm_classes)]
                        
                        synthetic = self.generate_single(
                            clean_cube=clean_cube,
                            fm_class=fm_class
                        )
                        
                        # Save to HDF5
                        grp = hf_out.create_group(f"sample_{sample_idx:06d}")
                        
                        grp.create_dataset(
                            'hsi_cube',
                            data=synthetic['cube'],
                            compression='gzip',
                            compression_opts=4
                        )
                        
                        grp.create_dataset(
                            'mask',
                            data=synthetic['mask'],
                            compression='gzip'
                        )
                        
                        # Metadata
                        grp.attrs['fm_class'] = synthetic['fm_class']
                        grp.attrs['n_objects'] = synthetic['n_objects']
                        grp.attrs['source_clean_idx'] = clean_idx
                        grp.attrs['objects_info'] = json.dumps(synthetic['objects'])
                        
                        stats[fm_class] += 1
                        sample_idx += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate synthetic {sample_idx}: {e}")
                        continue
        
        logger.info(f"Generated {sample_idx} synthetic samples")
        logger.info(f"Distribution: {stats}")
        
        # Save metadata
        metadata_path = Path(output_hdf5).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'n_samples': sample_idx,
                'n_clean_sources': n_clean,
                'n_synthetic_per_clean': n_synthetic_per_clean,
                'fm_class_distribution': stats,
                'size_range': self.size_range,
                'n_objects_range': self.n_objects_range
            }, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Example usage"""
    import yaml
    from ..signature_extraction.fm_signature_extractor import FMSignatureExtractor
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load FM signatures
    extractor = FMSignatureExtractor()
    fm_signatures = extractor.load_signatures(
        'data/processed/spectral_signatures/fm_signatures.npz'
    )
    
    # Initialize generator
    generator = SyntheticFMGenerator(
        fm_signatures=fm_signatures,
        insertion_strategy='realistic',
        size_range=tuple(config['synthetic']['fm_size_range']),
        n_objects_range=tuple(config['synthetic']['n_objects_per_cube']),
        gmm_aware=config['synthetic']['gmm_aware']
    )
    
    # Generate training set
    generator.generate_dataset(
        clean_cubes_hdf5='data/processed/harmonized_204bands/hsifood_processed.h5',
        output_hdf5='data/synthetic/train/synthetic_train.h5',
        n_synthetic_per_clean=config['synthetic']['n_synthetic_per_clean'],
        max_clean_samples=1000
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
