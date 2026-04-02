"""
Foreign Material Signature Extractor - WITH LAYER MAPPING
Extract real FM spectral signatures from AgriFoodAnomaly dataset
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import logging
import json
import re

logger = logging.getLogger(__name__)


class FMSignatureExtractor:
    """Extract and store FM spectral signatures with layer-to-class mapping"""
    
    def __init__(self, fm_classes: Optional[List[str]] = None):
        """
        Args:
            fm_classes: List of FM class names
        """
        self.fm_classes = fm_classes or [
            'textile', 'plastic', 'paper', 'metal', 'mineral'
        ]
        
        self.signatures = {fm_class: [] for fm_class in self.fm_classes}
        
        # Layer mapping based on AgriFoodAnomaly dataset structure
        # From README: Textile(40), Plastic(20), Paper(5), Metal(5), Mineral/wood/glass(5)
        self.layer_to_class = {
            # Textile and fiber-based materials (L1-L8, 40 samples)
            1: 'textile',
            2: 'textile', 
            3: 'textile',
            4: 'textile',
            5: 'textile',
            6: 'textile',
            7: 'textile',
            8: 'textile',
            # Plastics (L9-L11, 20 samples)
            9: 'plastic',
            10: 'plastic',
            11: 'plastic',
            # Paper-based (L12, 5 samples)
            12: 'paper',
            # Metals (L13, 5 samples)
            13: 'metal',
            # Wood/minerals/glass (L14-L15, 5 samples)
            14: 'mineral',
            15: 'mineral',
        }
        
        logger.info(f"FMSignatureExtractor initialized with classes: {self.fm_classes}")
    
    def extract_from_hdf5(self, hdf5_path: str, 
                         split: str = 'train',
                         min_pixels: int = 10):
        """
        Extract FM signatures from processed AgriFoodAnomaly dataset
        
        Args:
            hdf5_path: Path to processed HDF5 file
            split: Dataset split ('train', 'val', 'test')
            min_pixels: Minimum number of FM pixels required
        """
        logger.info(f"Extracting FM signatures from {hdf5_path}")
        
        extracted_count = 0
        skipped_no_mask = 0
        skipped_no_fm = 0
        skipped_few_pixels = 0
        
        with h5py.File(hdf5_path, 'r') as hf:
            sample_keys = [k for k in hf.keys() if k.startswith('sample_')]
            
            for sample_key in tqdm(sample_keys, desc=f"Extracting {split}"):
                grp = hf[sample_key]
                
                # Load cube
                cube = grp['hsi_cube'][:]
                
                # Check for mask
                if 'mask' not in grp:
                    skipped_no_mask += 1
                    continue
                
                mask = grp['mask'][:].astype(bool)
                
                # Check if mask has FM pixels
                n_fm_pixels = np.sum(mask)
                if n_fm_pixels == 0:
                    skipped_no_fm += 1
                    continue
                
                if n_fm_pixels < min_pixels:
                    skipped_few_pixels += 1
                    continue
                
                # Extract FM pixels
                fm_pixels = cube[mask]
                
                # Get FM class from sample name
                sample_name = grp.attrs.get('sample_name', '')
                fm_class = self._get_fm_class(sample_name)
                
                if fm_class and fm_class in self.signatures:
                    self.signatures[fm_class].append(fm_pixels)
                    extracted_count += 1
                    logger.debug(f"{sample_key} ({sample_name}) → {fm_class}: {n_fm_pixels} pixels")
                else:
                    logger.warning(f"Unknown class for {sample_key} ({sample_name})")
        
        logger.info(f"Extraction complete:")
        logger.info(f"  Extracted: {extracted_count} samples")
        logger.info(f"  Skipped - no mask: {skipped_no_mask}")
        logger.info(f"  Skipped - no FM pixels: {skipped_no_fm}")
        logger.info(f"  Skipped - too few pixels: {skipped_few_pixels}")
        
        self._print_statistics()
    
    def _get_fm_class(self, sample_name: str) -> Optional[str]:
        """
        Get FM class from sample name by extracting layer number
        
        Sample format: UseCase_1_(Avoine1)_Anomaly_Easy_L10_1
        Extract L10 → layer 10 → plastic
        
        Args:
            sample_name: Sample name from attributes
            
        Returns:
            fm_class: FM class name or None
        """
        # Extract layer number using regex
        match = re.search(r'_L(\d+)_', sample_name)
        
        if match:
            layer_num = int(match.group(1))
            fm_class = self.layer_to_class.get(layer_num, None)
            
            if fm_class:
                return fm_class
            else:
                logger.debug(f"Layer {layer_num} not in mapping for {sample_name}")
                return None
        
        logger.warning(f"Could not extract layer number from: {sample_name}")
        return None
    
    def compute_statistics(self) -> Dict:
        """Compute statistical summaries for each FM class"""
        logger.info("Computing spectral statistics...")
        
        stats = {}
        
        for fm_class in self.fm_classes:
            if not self.signatures[fm_class]:
                logger.warning(f"No signatures for class: {fm_class}")
                continue
            
            # Concatenate all pixels for this class
            all_pixels = np.vstack(self.signatures[fm_class])
            
            stats[fm_class] = {
                'mean': np.mean(all_pixels, axis=0),
                'std': np.std(all_pixels, axis=0),
                'median': np.median(all_pixels, axis=0),
                'p25': np.percentile(all_pixels, 25, axis=0),
                'p75': np.percentile(all_pixels, 75, axis=0),
                'min': np.min(all_pixels, axis=0),
                'max': np.max(all_pixels, axis=0),
                'n_samples': len(self.signatures[fm_class]),
                'n_pixels': len(all_pixels)
            }
            
            logger.info(f"{fm_class:10s}: {stats[fm_class]['n_pixels']:>8,} pixels "
                       f"from {stats[fm_class]['n_samples']:>3} samples")
        
        return stats
    
    def save_signatures(self, output_path: str):
        """Save FM signatures to file"""
        stats = self.compute_statistics()
        
        if not stats:
            logger.error("No signatures to save!")
            # Save empty file to avoid errors downstream
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **{})
            
            # Save empty metadata
            metadata_path = Path(output_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({}, f, indent=2)
            return
        
        # Prepare data for saving
        save_dict = {}
        
        for fm_class, class_stats in stats.items():
            for stat_name, stat_value in class_stats.items():
                if isinstance(stat_value, np.ndarray):
                    save_dict[f"{fm_class}_{stat_name}"] = stat_value.astype(np.float32)
                else:
                    save_dict[f"{fm_class}_{stat_name}"] = stat_value
        
        # Save as NPZ
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path, **save_dict)
        
        logger.info(f"Signatures saved to {output_path}")
        
        # Save metadata as JSON
        metadata_path = output_file.with_suffix('.json')
        metadata = {
            fm_class: {
                'n_samples': int(stats[fm_class]['n_samples']),
                'n_pixels': int(stats[fm_class]['n_pixels'])
            }
            for fm_class in stats.keys()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_signatures(self, npz_path: str) -> Dict:
        """Load FM signatures from file"""
        logger.info(f"Loading signatures from {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        signatures = {}
        
        # Get all unique class names from the file
        class_names = set()
        for key in data.keys():
            if key.endswith('_mean'):
                class_name = key.replace('_mean', '')
                class_names.add(class_name)
        
        for fm_class in class_names:
            if f"{fm_class}_mean" in data:
                signatures[fm_class] = {
                    'mean': data[f"{fm_class}_mean"],
                    'std': data[f"{fm_class}_std"],
                    'median': data[f"{fm_class}_median"],
                    'p25': data[f"{fm_class}_p25"],
                    'p75': data[f"{fm_class}_p75"],
                    'min': data[f"{fm_class}_min"],
                    'max': data[f"{fm_class}_max"],
                    'n_samples': int(data[f"{fm_class}_n_samples"]),
                    'n_pixels': int(data[f"{fm_class}_n_pixels"])
                }
        
        logger.info(f"Loaded signatures for {len(signatures)} classes")
        
        return signatures
    
    def _print_statistics(self):
        """Print extraction statistics"""
        logger.info("\n" + "="*50)
        logger.info("FM Signature Extraction Statistics")
        logger.info("="*50)
        
        for fm_class in self.fm_classes:
            n_samples = len(self.signatures[fm_class])
            if n_samples > 0:
                total_pixels = sum(len(pixels) for pixels in self.signatures[fm_class])
                logger.info(f"{fm_class:10s}: {n_samples:>3} samples, {total_pixels:>8,} pixels")
            else:
                logger.info(f"{fm_class:10s}: No signatures extracted")
        
        logger.info("="*50 + "\n")


def main():
    """Example usage"""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    extractor = FMSignatureExtractor(
        fm_classes=['textile', 'plastic', 'paper', 'metal', 'mineral']
    )
    
    # Extract from training set
    extractor.extract_from_hdf5(
        hdf5_path='data/processed/harmonized_204bands/agrifood_train_processed.h5',
        split='train',
        min_pixels=10
    )
    
    # Save signatures
    extractor.save_signatures(
        output_path='data/processed/spectral_signatures/fm_signatures.npz'
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
