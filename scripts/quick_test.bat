@echo off
REM Quick sanity test - verify model loads and runs inference

echo ==========================================
echo Quick Test
echo ==========================================

call conda activate hsi-fm

python -c "
import torch
from src.models.litenet import LiteNet

print('[TEST 1] Loading model architecture...')
model = LiteNet(n_bands=204, spatial_channels=3,
                spectral_output_dim=128, spatial_output_dim=64,
                fusion_hidden_dim=64, dropout=0.3,
                spectral_architecture='simple', spatial_input_size=3)
print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
print('[OK] Model architecture OK')

print('[TEST 2] Forward pass...')
spec = torch.randn(4, 204)
spat = torch.randn(4, 3, 3, 3)
out = model(spec, spat)
print(f'  Output shape: {tuple(out.shape)}')
print('[OK] Forward pass OK')

print('[TEST 3] Loading trained model...')
import os
if os.path.exists('experiments/phase2_real/best_model.pth'):
    ckpt = torch.load('experiments/phase2_real/best_model.pth',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print('[OK] Phase 2 model loaded OK')
elif os.path.exists('experiments/phase1_synthetic/best_model.pth'):
    ckpt = torch.load('experiments/phase1_synthetic/best_model.pth',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print('[OK] Phase 1 model loaded OK')
else:
    print('[SKIP] No trained model found - run run_training.bat first')
print()
print('All tests passed!')
"
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Quick test failed
    exit /b 1
)
echo.
echo [OK] Quick test passed!
