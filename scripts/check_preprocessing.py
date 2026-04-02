import sys
import os
import torch
from pathlib import Path

# Force UTF-8 output to avoid CP1252 encoding errors on Windows CMD
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FILES = {
    "Synthetic Train": "data/synthetic/train_pt/synthetic_train_preprocessed.pt",
    "Synthetic Val":   "data/synthetic/val_pt/synthetic_val_preprocessed.pt",
    "Real Train":      "data/processed/harmonized_204bands/agrifood_train_preprocessed.pt",
    "Real Val":        "data/processed/harmonized_204bands/agrifood_val_preprocessed.pt",
}

def check_pt_file(path, name):
    try:
        if not Path(path).exists():
            print(f"[MISSING] {name}: file not found at {path}")
            return False

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        data    = torch.load(path, map_location="cpu", weights_only=False)

        labels  = data["labels"]
        n       = len(labels)
        n_pos   = int((labels == 1).sum())
        n_neg   = int((labels == 0).sum())
        ratio   = 100.0 * n_pos / max(n, 1)

        print(f"[OK] {name}: {size_mb:.1f} MB, {n:,} samples "
              f"(pos={n_pos:,}, neg={n_neg:,}, ratio={ratio:.2f}%)")
        return True

    except Exception as e:
        print(f"[FAIL] {name}: CORRUPT ({e})")
        return False


def main():
    print("=" * 70)
    print("Preprocessing Status Check")
    print("=" * 70)

    all_ok = True
    for name, path in FILES.items():
        ok = check_pt_file(path, name)
        if not ok:
            all_ok = False

    print("=" * 70)

    if all_ok:
        print("[OK] All files OK. Ready for training.")
        sys.exit(0)
    else:
        print("[FAIL] Some files are missing or corrupt.")
        sys.exit(1)


if __name__ == "__main__":
    main()
