"""
HSI-AgriFoodAnomaly Dataset Extractor
Extracts train/val/test splits with masks
"""

import zipfile
from pathlib import Path
import numpy as np
import h5py
import spectral as spy
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgriFoodExtractor:
    """Extract HSI-AgriFoodAnomaly dataset"""
    
    def __init__(self, raw_dir="data/raw/HSI-AgriFoodAnomaly",
                 output_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.splits = {
            'train': 'train_UseCase_1_(Avoine1).zip',
            'val': 'val_UseCase_1_(Avoine1).zip',
            'test': 'test_UseCase_1_(Avoine1).zip'
        }
    
    def extract_split(self, split_name):
        """Extract a specific split (train/val/test)"""
        zip_name = self.splits[split_name]
        zip_path = self.raw_dir / zip_name
        
        if not zip_path.exists():
            logger.error(f"Split file not found: {zip_path}")
            return None
        
        logger.info(f"Extracting {split_name} split...")
        
        extract_dir = self.raw_dir / f"extracted_{split_name}"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        return extract_dir
    
    def scan_split_data(self, extract_dir):
        """Scan for HSI cubes and masks in extracted split"""
        # Look for .bil files (hyperspectral) and .png files (masks)
        bil_files = list(extract_dir.rglob("*.bil"))
        
        samples = []
        for bil_path in bil_files:
            # Find corresponding .hdr and mask
            hdr_path = bil_path.with_suffix('.bil.hdr')
            
            # Look for mask (usually has same name with _mask or in masks folder)
            mask_path = None
            possible_mask_names = [
                bil_path.with_suffix('.png'),
                bil_path.parent / f"{bil_path.stem}_mask.png",
                bil_path.parent / "masks" / f"{bil_path.stem}.png"
            ]
            
            for mask_candidate in possible_mask_names:
                if mask_candidate.exists():
                    mask_path = mask_candidate
                    break
            
            if hdr_path.exists():
                samples.append({
                    'bil': bil_path,
                    'hdr': hdr_path,
                    'mask': mask_path
                })
        
        logger.info(f"Found {len(samples)} samples")
        return samples
    
    def read_bil_cube(self, hdr_path):
        """Read BIL format HSI cube"""
        try:
            img = spy.open_image(str(hdr_path))
            cube = img.load()
            
            metadata = {
                'wavelengths': img.metadata.get('wavelength', None),
                'bands': img.nbands,
                'lines': img.nrows,
                'samples': img.ncols,
                'interleave': 'BIL'
            }
            
            return cube, metadata
        except Exception as e:
            logger.error(f"Failed to read {hdr_path}: {e}")
            return None, None
    
    def read_mask(self, mask_path):
        """Read mask PNG file"""
        if mask_path is None or not mask_path.exists():
            return None
        
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            # Convert to binary (0 = clean, 1 = FM)
            mask = (mask > 127).astype(np.uint8)
            return mask
        except Exception as e:
            logger.error(f"Failed to read mask {mask_path}: {e}")
            return None
    
    def save_split_to_hdf5(self, samples, output_path):
        """Save split data to HDF5"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(samples)} samples to {output_path}")
        
        with h5py.File(output_path, 'w') as hf:
            for idx, sample in enumerate(tqdm(samples, desc="Saving to HDF5")):
                cube, metadata = self.read_bil_cube(sample['hdr'])
                
                if cube is None:
                    continue
                
                mask = self.read_mask(sample['mask'])
                
                # Create group
                grp = hf.create_group(f"sample_{idx:04d}")
                
                # Save cube
                grp.create_dataset(
                    'hsi_cube',
                    data=cube.astype(np.float32),
                    compression='gzip',
                    compression_opts=4
                )
                
                # Save mask if available
                if mask is not None:
                    grp.create_dataset('mask', data=mask)
                    grp.attrs['has_mask'] = True
                else:
                    grp.attrs['has_mask'] = False
                
                # Save metadata
                for key, value in metadata.items():
                    if value is not None and isinstance(value, (str, int, float)):
                        grp.attrs[key] = value
                
                grp.attrs['source_file'] = str(sample['bil'])
        
        logger.info(f"Saved to {output_path}")
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
    
    def run(self):
        """Run extraction for all splits"""
        logger.info("="*60)
        logger.info("HSI-AgriFoodAnomaly Extraction Pipeline")
        logger.info("="*60)
        
        for split_name in ['train', 'val', 'test']:
            logger.info(f"\nProcessing {split_name} split...")
            
            # Extract split
            extract_dir = self.extract_split(split_name)
            if extract_dir is None:
                continue
            
            # Scan for data
            samples = self.scan_split_data(extract_dir)
            
            if not samples:
                logger.warning(f"No samples found in {split_name} split")
                continue
            
            # Save to HDF5
            output_path = self.output_dir / f"agrifood_{split_name}.h5"
            self.save_split_to_hdf5(samples, output_path)
        
        logger.info("="*60)
        logger.info("Extraction complete!")
        logger.info("="*60)


def main():
    """Main entry point"""
    extractor = AgriFoodExtractor(
        raw_dir="data/raw/HSI-AgriFoodAnomaly",
        output_dir="data/processed"
    )
    extractor.run()


if __name__ == "__main__":
    main()
