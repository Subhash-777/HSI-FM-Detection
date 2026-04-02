@echo off
REM Export trained model to ONNX / TorchScript

echo ==========================================
echo Export Model
echo ==========================================

call conda activate hsi-fm

set MODEL_PATH=experiments\phase2_real\best_model.pth
set EXPORT_DIR=outputs\exported_models
mkdir "%EXPORT_DIR%" 2>nul

if not exist "%MODEL_PATH%" (
    echo [ERROR] Trained model not found: %MODEL_PATH%
    echo Run run_training.bat first.
    exit /b 1
)

echo [INFO] Exporting model from: %MODEL_PATH%
echo [INFO] Export directory: %EXPORT_DIR%
echo.

python -c "
import torch
from src.models.litenet import LiteNet

device = 'cpu'
model = LiteNet(n_bands=204, spatial_channels=3,
                spectral_output_dim=128, spatial_output_dim=64,
                fusion_hidden_dim=64, dropout=0.3,
                spectral_architecture='simple', spatial_input_size=3)

ckpt = torch.load('experiments/phase2_real/best_model.pth',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Export TorchScript
spec_dummy = torch.randn(1, 204)
spat_dummy = torch.randn(1, 3, 3, 3)
traced = torch.jit.trace(model, (spec_dummy, spat_dummy))
traced.save('outputs/exported_models/litenet_traced.pt')
print('[OK] TorchScript saved: outputs/exported_models/litenet_traced.pt')

# Export ONNX
torch.onnx.export(
    model, (spec_dummy, spat_dummy),
    'outputs/exported_models/litenet.onnx',
    input_names=['spectrum', 'spatial'],
    output_names=['logits'],
    dynamic_axes={'spectrum': {0: 'batch'}, 'spatial': {0: 'batch'}},
    opset_version=17
)
print('[OK] ONNX saved: outputs/exported_models/litenet.onnx')

# Save metadata
import json
meta = {
    'n_bands': 204, 'spatial_size': 3,
    'params': sum(p.numel() for p in model.parameters()),
    'epoch': ckpt.get('epoch', 'unknown'),
    'val_loss': ckpt.get('score', 'unknown')
}
with open('outputs/exported_models/model_info.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('[OK] Metadata saved: outputs/exported_models/model_info.json')
"
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Export failed
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Model export complete!
echo     outputs\exported_models\
echo       litenet_traced.pt   (TorchScript)
echo       litenet.onnx        (ONNX)
echo       model_info.json     (Metadata)
echo ==========================================
