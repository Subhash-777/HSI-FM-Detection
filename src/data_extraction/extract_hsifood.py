"""
HSIFoodIngr-64 Dataset Extractor (Improved)
Extracts .tar.gz files containing ENVI format HSI cubes
Includes ingredient labels from JSON files
"""

import os
import tarfile
import zipfile
import json
from pathlib import Path
import numpy as np
import h5py
import spectral as spy
from tqdm import tqdm
import logging
import gc
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HSIFoodExtractor:
    """Extract HSIFoodIngr-64 dataset with improved error handling"""
    
    def __init__(self, raw_dir="data/raw/HSIFoodIngr-64", 
                 output_dir="data/processed",
                 keep_extracted=False):
        """
        Args:
            raw_dir: Directory containing raw .tar.gz files
            output_dir: Directory for processed HDF5 output
            keep_extracted: Keep extracted intermediate files (False saves disk space)
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_extracted = keep_extracted
        
        # Statistics
        self.stats = {
            'tar_files_processed': 0,
            'zip_files_processed': 0,
            'hsi_cubes_found': 0,
            'hsi_cubes_saved': 0,
            'errors': []
        }
    
    def extract_tar_gz_files(self) -> Path:
        """
        Extract all .tar.gz files
        Returns: Path to extraction directory
        """
        logger.info("Step 1/4: Extracting .tar.gz files...")
        
        tar_files = sorted(self.raw_dir.glob("*.tar.gz"))
        logger.info(f"Found {len(tar_files)} .tar.gz files")
        
        if not tar_files:
            raise FileNotFoundError(f"No .tar.gz files found in {self.raw_dir}")
        
        extract_dir = self.raw_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        for tar_path in tqdm(tar_files, desc="Extracting .tar.gz"):
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    # Extract safely (prevent path traversal attacks)
                    tar.extractall(extract_dir, filter='data')
                    self.stats['tar_files_processed'] += 1
            except Exception as e:
                error_msg = f"Failed to extract {tar_path.name}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        logger.info(f"Extracted {self.stats['tar_files_processed']}/{len(tar_files)} .tar.gz files")
        return extract_dir
    
    def extract_zip_files(self, extract_dir: Path) -> Path:
        """
        Extract all .zip files from extracted .tar.gz
        Returns: Path to final HSI cubes directory
        """
        logger.info("Step 2/4: Extracting .zip files...")
        
        zip_files = sorted(extract_dir.glob("*.zip"))
        logger.info(f"Found {len(zip_files)} .zip files")
        
        if not zip_files:
            logger.warning("No .zip files found, checking if already extracted...")
            # Maybe files are already extracted
            return extract_dir
        
        final_dir = extract_dir / "hsi_cubes"
        final_dir.mkdir(exist_ok=True)
        
        for zip_path in tqdm(zip_files, desc="Extracting .zip"):
            try:
                # Create subdirectory for each zip
                subdir = final_dir / zip_path.stem
                subdir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Check for malicious files
                    for name in zf.namelist():
                        if name.startswith('/') or '..' in name:
                            logger.warning(f"Suspicious file path: {name}, skipping")
                            continue
                    zf.extractall(subdir)
                    self.stats['zip_files_processed'] += 1
                
                # Optionally delete .zip after extraction to save space
                if not self.keep_extracted:
                    zip_path.unlink()
                    
            except Exception as e:
                error_msg = f"Failed to extract {zip_path.name}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        logger.info(f"Extracted {self.stats['zip_files_processed']}/{len(zip_files)} .zip files")
        return final_dir
    
    def scan_envi_files(self, hsi_dir: Path) -> list:
        """
        Scan for ENVI format HSI files (.hdr + .dat pairs) and JSON labels
        Returns: List of tuples (hdr_path, dat_path, json_path)
        """
        logger.info("Step 3/4: Scanning for ENVI files...")
        
        hdr_files = list(hsi_dir.rglob("*.hdr"))
        logger.info(f"Found {len(hdr_files)} .hdr files")
        
        samples = []
        
        for hdr_path in hdr_files:
            # Skip if it's actually a .bil.hdr file
            if hdr_path.stem.endswith('.bil'):
                continue
            
            # Find corresponding .dat file
            dat_path = hdr_path.with_suffix('.dat')
            if not dat_path.exists():
                # Try without extension
                dat_path = hdr_path.with_name(hdr_path.stem)
                if not dat_path.exists():
                    logger.warning(f"Missing .dat for {hdr_path.name}")
                    continue
            
            # Find corresponding .json label file (optional)
            json_path = hdr_path.with_suffix('.json')
            if not json_path.exists():
                json_path = None
            
            samples.append({
                'hdr': hdr_path,
                'dat': dat_path,
                'json': json_path,
                'name': hdr_path.stem
            })
        
        self.stats['hsi_cubes_found'] = len(samples)
        logger.info(f"Found {len(samples)} complete ENVI sample pairs")
        logger.info(f"  {sum(1 for s in samples if s['json'])} with JSON labels")
        
        return samples
    
    def read_envi_cube(self, hdr_path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Read ENVI format HSI cube
        Returns: (cube array, metadata dict) or (None, None) on error
        """
        try:
            img = spy.open_image(str(hdr_path))
            cube = img.load()
            
            # Parse wavelengths
            wavelengths = img.metadata.get('wavelength', None)
            if wavelengths and isinstance(wavelengths, list):
                wavelengths = [float(w) for w in wavelengths]
            
            metadata = {
                'bands': img.nbands,
                'lines': img.nrows,
                'samples': img.ncols,
                'data_type': img.dtype.name,
                'interleave': img.metadata.get('interleave', 'unknown'),
                'source_file': str(hdr_path)
            }
            
            # Add wavelengths separately (as array)
            if wavelengths:
                metadata['wavelengths'] = wavelengths
            
            return cube, metadata
            
        except Exception as e:
            logger.error(f"Failed to read {hdr_path.name}: {e}")
            return None, None
    
    def read_json_labels(self, json_path: Optional[Path]) -> Optional[Dict]:
        """
        Read ingredient labels from JSON file
        Returns: Dictionary with dish and ingredient info, or None
        """
        if json_path is None or not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract relevant information
            labels = {
                'dish': None,
                'ingredients': [],
                'num_ingredients': 0
            }
            
            # Parse label structure (adjust based on actual JSON format)
            if 'label' in data:
                label = data['label']
                if '/' in label:
                    parts = label.split('/')
                    labels['dish'] = parts[0]
                    if len(parts) > 1:
                        labels['ingredients'] = [parts[1]]
            
            if 'ingredients' in data:
                labels['ingredients'] = data['ingredients']
            
            labels['num_ingredients'] = len(labels['ingredients'])
            
            return labels
            
        except Exception as e:
            logger.warning(f"Failed to read JSON {json_path.name}: {e}")
            return None
    
    def save_to_hdf5(self, samples: list, output_path: Path, max_samples: Optional[int] = None):
        """
        Save HSI cubes to HDF5 format with improved structure
        """
        if max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"Step 4/4: Saving {len(samples)} cubes to {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        with h5py.File(output_path, 'w') as hf:
            # Create metadata group
            meta_group = hf.create_group('metadata')
            meta_group.attrs['dataset'] = 'HSIFoodIngr-64'
            meta_group.attrs['total_samples'] = len(samples)
            meta_group.attrs['description'] = 'Clean food hyperspectral images'
            
            for idx, sample in enumerate(tqdm(samples, desc="Saving to HDF5")):
                cube, metadata = self.read_envi_cube(sample['hdr'])
                
                if cube is None:
                    continue
                
                # Read labels if available
                labels = self.read_json_labels(sample['json'])
                
                # Create group for this sample
                grp = hf.create_group(f"sample_{idx:04d}")
                
                # Save cube with compression
                grp.create_dataset(
                    'hsi_cube',
                    data=cube.astype(np.float32),
                    compression='gzip',
                    compression_opts=4
                )
                
                # Save basic metadata as attributes
                for key, value in metadata.items():
                    if key != 'wavelengths' and isinstance(value, (str, int, float)):
                        grp.attrs[key] = value
                
                # Save wavelengths as dataset (if available)
                if 'wavelengths' in metadata and metadata['wavelengths']:
                    grp.create_dataset('wavelengths', data=metadata['wavelengths'])
                
                # Save labels (if available)
                if labels:
                    grp.attrs['has_labels'] = True
                    if labels['dish']:
                        grp.attrs['dish'] = labels['dish']
                    if labels['ingredients']:
                        grp.attrs['ingredients'] = ','.join(labels['ingredients'])
                    grp.attrs['num_ingredients'] = labels['num_ingredients']
                else:
                    grp.attrs['has_labels'] = False
                
                # Add sample identification
                grp.attrs['sample_name'] = sample['name']
                grp.attrs['sample_index'] = idx
                
                saved_count += 1
                
                # Periodic garbage collection for large datasets
                if idx % 100 == 0:
                    gc.collect()
        
        self.stats['hsi_cubes_saved'] = saved_count
        
        # Report statistics
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {saved_count}/{len(samples)} samples to {output_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
    
    def cleanup_extracted_files(self):
        """Remove intermediate extracted files to save disk space"""
        if not self.keep_extracted:
            logger.info("Cleaning up intermediate files...")
            extract_dir = self.raw_dir / "extracted"
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir)
                logger.info("Cleanup complete")
    
    def print_statistics(self):
        """Print extraction statistics"""
        logger.info("\n" + "="*60)
        logger.info("Extraction Statistics")
        logger.info("="*60)
        logger.info(f".tar.gz files processed: {self.stats['tar_files_processed']}")
        logger.info(f".zip files processed: {self.stats['zip_files_processed']}")
        logger.info(f"HSI cubes found: {self.stats['hsi_cubes_found']}")
        logger.info(f"HSI cubes saved: {self.stats['hsi_cubes_saved']}")
        
        if self.stats['errors']:
            logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        else:
            logger.info("No errors encountered")
        
        logger.info("="*60)
    
    def run(self, max_samples: Optional[int] = None):
        """
        Run complete extraction pipeline
        
        Args:
            max_samples: Limit number of samples to process (None = all)
        """
        logger.info("="*60)
        logger.info("HSIFoodIngr-64 Extraction Pipeline (Improved)")
        logger.info("="*60)
        
        try:
            # Step 1: Extract .tar.gz files
            extract_dir = self.extract_tar_gz_files()
            
            # Step 2: Extract .zip files
            hsi_dir = self.extract_zip_files(extract_dir)
            
            # Step 3: Scan for ENVI files
            samples = self.scan_envi_files(hsi_dir)
            
            if not samples:
                logger.error("No ENVI files found!")
                return
            
            # Step 4: Save to HDF5
            output_path = self.output_dir / "hsifood_clean_samples.h5"
            self.save_to_hdf5(samples, output_path, max_samples)
            
            # Step 5: Cleanup (optional)
            if not self.keep_extracted:
                self.cleanup_extracted_files()
            
            # Print statistics
            self.print_statistics()
            
            logger.info("="*60)
            logger.info("✅ Extraction complete!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Fatal error during extraction: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point"""
    extractor = HSIFoodExtractor(
        raw_dir="data/raw/HSIFoodIngr-64",
        output_dir="data/processed",
        keep_extracted=False  # Set to True to keep intermediate files
    )
    
    # Extract all samples (or set max_samples for testing)
    extractor.run(max_samples=None)  # Set to 10 for quick test


if __name__ == "__main__":
    main()
