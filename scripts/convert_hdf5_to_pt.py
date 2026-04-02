"""
Memory-safe HDF5 → PyTorch converter
Saves incrementally to avoid OOM on large datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_choice(arr, k, rng):
    if len(arr) == 0 or k <= 0:
        return np.empty((0, 2), dtype=np.int64)
    k = min(k, len(arr))
    idx = rng.choice(len(arr), size=k, replace=False)
    return arr[idx]


def convert_dataset_batched(
    hdf5_path: str,
    pca_path: str,
    output_dir: str,
    samples_per_cube: int = 10,
    batch_size: int = 50,
    pos_per_cube: int = 50,
    neg_per_cube: int = 200,
    seed: int = 42,
    save_every: int = 5000,  # NEW: save checkpoint every N samples
):
    """
    Convert with incremental saving to avoid OOM
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("Okayy HDF5 to PyTorch Tensor Conversion (MEMORY-SAFE)")
        logger.info("=" * 70)
        logger.info(f"Input HDF5: {hdf5_path}")
        logger.info(f"PCA model:  {pca_path}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"pos_per_cube={pos_per_cube}, neg_per_cube={neg_per_cube}")
        logger.info(f"save_every={save_every} samples")

        rng = np.random.default_rng(seed)

        logger.info("Loading PCA model...")
        pca = joblib.load(pca_path)

        logger.info("Opening HDF5 file...")
        hf = h5py.File(hdf5_path, "r", swmr=True, libver="latest")
        sample_keys = [k for k in hf.keys() if k.startswith("sample_")]
        total_cubes = len(sample_keys)

        if total_cubes == 0:
            raise RuntimeError(f"No sample_* groups found in {hdf5_path}")

        logger.info(f"✓ Found {total_cubes} cubes")
        logger.info(f"Batch size: {batch_size} cubes")

        # Temporary storage (will be saved/cleared periodically)
        all_spectra = []
        all_spatial = []
        all_labels = []
        
        # Final outputs (will accumulate chunk file paths)
        chunk_files = []
        chunk_idx = 0
        total_samples = 0

        n_batches = (total_cubes + batch_size - 1) // batch_size

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, total_cubes)
            keys = sample_keys[start:end]

            logger.info(f"\nBatch {b+1}/{n_batches}: cubes {start}-{end}")
            
            for key in tqdm(keys, desc="Processing", ncols=90):
                grp = hf[key]
                cube = grp["hsi_cube"][:]
                mask = grp["mask"][:]

                H, W, C = cube.shape
                mask_bin = (mask > 0).astype(np.uint8)

                pos_idx = np.argwhere(mask_bin == 1)
                neg_idx = np.argwhere(mask_bin == 0)

                chosen_pos = _safe_choice(pos_idx, pos_per_cube, rng)
                chosen_neg = _safe_choice(neg_idx, neg_per_cube, rng)

                chosen = np.concatenate([chosen_pos, chosen_neg], axis=0)
                rng.shuffle(chosen)

                for (row, col) in chosen:
                    spectrum = cube[row, col, :].astype(np.float32)
                    label = 1.0 if mask_bin[row, col] == 1 else 0.0

                    r0, r1 = max(0, row - 1), min(H, row + 2)
                    c0, c1 = max(0, col - 1), min(W, col + 2)
                    patch = cube[r0:r1, c0:c1, :]

                    patch_flat = patch.reshape(-1, C)
                    patch_pca = pca.transform(patch_flat).reshape(
                        patch.shape[0], patch.shape[1], 3
                    ).astype(np.float32)

                    if patch_pca.shape[0] < 3 or patch_pca.shape[1] < 3:
                        padded = np.zeros((3, 3, 3), dtype=np.float32)
                        padded[:patch_pca.shape[0], :patch_pca.shape[1], :] = patch_pca
                        patch_pca = padded

                    all_spectra.append(spectrum)
                    all_spatial.append(patch_pca.transpose(2, 0, 1))
                    all_labels.append(label)

                # Save checkpoint if buffer is large
                if len(all_labels) >= save_every:
                    chunk_file = output_dir / f"_chunk_{chunk_idx:04d}.pt"
                    _save_chunk(chunk_file, all_spectra, all_spatial, all_labels)
                    chunk_files.append(chunk_file)
                    total_samples += len(all_labels)
                    logger.info(f"  💾 Saved chunk {chunk_idx} ({len(all_labels):,} samples)")
                    
                    # Clear buffer
                    all_spectra.clear()
                    all_spatial.clear()
                    all_labels.clear()
                    chunk_idx += 1

        hf.close()

        # Save remaining samples
        if len(all_labels) > 0:
            chunk_file = output_dir / f"_chunk_{chunk_idx:04d}.pt"
            _save_chunk(chunk_file, all_spectra, all_spatial, all_labels)
            chunk_files.append(chunk_file)
            total_samples += len(all_labels)
            logger.info(f"  💾 Saved final chunk {chunk_idx} ({len(all_labels):,} samples)")

        # Merge all chunks into final file
        logger.info("\nMerging chunks into final .pt file...")
        base_name = Path(hdf5_path).stem.replace("_processed", "")
        output_file = output_dir / f"{base_name}_preprocessed.pt"
        
        _merge_chunks(chunk_files, output_file)
        
        # Cleanup chunks
        for cf in chunk_files:
            cf.unlink()
        
        logger.info("=" * 70)
        logger.info("✓ CONVERSION COMPLETE")
        logger.info(f"  Total samples: {total_samples:,}")
        logger.info(f"  Output: {output_file}")
        logger.info("=" * 70)
        
        return str(output_file)

    except Exception as e:
        logger.error(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _save_chunk(path: Path, spectra_list, spatial_list, labels_list):
    """Save a chunk of samples"""
    torch.save(
        {
            "spectra": torch.tensor(np.array(spectra_list, dtype=np.float32)),
            "spatial": torch.tensor(np.array(spatial_list, dtype=np.float32)),
            "labels": torch.tensor(np.array(labels_list, dtype=np.float32)),
            "n_samples": len(labels_list),
        },
        path,
    )


def _merge_chunks(chunk_files: list, output_file: Path):
    """Merge chunk files into single .pt file"""
    all_spectra = []
    all_spatial = []
    all_labels = []
    
    for cf in tqdm(chunk_files, desc="Merging", ncols=80):
        data = torch.load(cf, map_location="cpu", weights_only=False)
        all_spectra.append(data["spectra"])
        all_spatial.append(data["spatial"])
        all_labels.append(data["labels"])
    
    spectra = torch.cat(all_spectra, dim=0)
    spatial = torch.cat(all_spatial, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    ratio = float(labels.mean().item()) if len(labels) else 0.0
    
    logger.info(f"✓ Label verification:")
    logger.info(f"  Unique labels: {torch.unique(labels).tolist()}")
    logger.info(f"  Neg: {n_neg:,} | Pos: {n_pos:,} | Pos ratio: {ratio:.2%}")
    
    if n_pos == 0:
        logger.warning("⚠ No positive samples!")
    
    torch.save(
        {
            "spectra": spectra,
            "spatial": spatial,
            "labels": labels,
            "n_samples": int(labels.numel()),
        },
        output_file,
    )
    
    size_mb = output_file.stat().st_size / (1024**2)
    logger.info(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--pca", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--pos-per-cube", type=int, default=50)
    ap.add_argument("--neg-per-cube", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N samples")
    args = ap.parse_args()

    convert_dataset_batched(
        hdf5_path=args.hdf5,
        pca_path=args.pca,
        output_dir=args.output,
        samples_per_cube=args.samples,
        batch_size=args.batch_size,
        pos_per_cube=args.pos_per_cube,
        neg_per_cube=args.neg_per_cube,
        seed=args.seed,
        save_every=args.save_every,
    )
