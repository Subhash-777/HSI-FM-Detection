"""
Verify training completed successfully
"""

import sys
from pathlib import Path
import torch
import json

def verify_training(checkpoint_dir='experiments/phase1_synthetic'):
    """Verify training artifacts"""
    
    checkpoint_dir = Path(checkpoint_dir)
    
    checks = {
        'checkpoint_exists': False,
        'metadata_exists': False,
        'model_loadable': False,
        'metrics_valid': False
    }
    
    # 1. Check best_model.pth exists
    model_path = checkpoint_dir / 'best_model.pth'
    if model_path.exists():
        checks['checkpoint_exists'] = True
        print(f"✓ Checkpoint found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ Checkpoint NOT found: {model_path}")
        return checks
    
    # 2. Check metadata
    metadata_path = checkpoint_dir / 'best_model_metadata.json'
    if metadata_path.exists():
        checks['metadata_exists'] = True
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"✓ Metadata found")
        print(f"  Epoch: {metadata.get('epoch', 'N/A')}")
        print(f"  Best score: {metadata.get('best_score', 'N/A'):.4f}")
    else:
        print(f"✗ Metadata NOT found: {metadata_path}")
    
    # 3. Try loading checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        required_keys = ['epoch', 'model_state_dict', 'score']
        
        if all(k in checkpoint for k in required_keys):
            checks['model_loadable'] = True
            print(f"✓ Checkpoint loadable")
            print(f"  Keys: {list(checkpoint.keys())}")
        else:
            print(f"✗ Checkpoint missing keys: {required_keys}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return checks
    
    # 4. Check metrics validity
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        if 'loss' in val_metrics and 'f1' in val_metrics:
            checks['metrics_valid'] = True
            print(f"✓ Validation metrics valid")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f}")
            print(f"  Precision: {val_metrics.get('precision', 'N/A'):.4f}")
            print(f"  Recall: {val_metrics.get('recall', 'N/A'):.4f}")
    
    # Summary
    print("\n" + "="*50)
    passed = sum(checks.values())
    total = len(checks)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("✓✓✓ TRAINING COMPLETED SUCCESSFULLY ✓✓✓")
        return True
    else:
        print("✗✗✗ TRAINING INCOMPLETE OR FAILED ✗✗✗")
        return False


if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'experiments/phase1_synthetic'
    success = verify_training(checkpoint_dir)
    sys.exit(0 if success else 1)
