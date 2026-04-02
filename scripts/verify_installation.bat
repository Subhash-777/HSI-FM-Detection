@echo off
REM Verify installation of all required packages

echo ==========================================
echo Verifying Installation
echo ==========================================

call conda activate hsi-fm

python -c "import torch; print('[OK] PyTorch:', torch.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] PyTorch not found

python -c "import torch; print('[OK] CUDA available:', torch.cuda.is_available())"

python -c "import numpy; print('[OK] NumPy:', numpy.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] NumPy not found

python -c "import scipy; print('[OK] SciPy:', scipy.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] SciPy not found

python -c "import sklearn; print('[OK] scikit-learn:', sklearn.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] scikit-learn not found

python -c "import h5py; print('[OK] h5py:', h5py.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] h5py not found

python -c "import yaml; print('[OK] PyYAML: OK')"
if %ERRORLEVEL% NEQ 0 echo [FAIL] PyYAML not found

python -c "import tqdm; print('[OK] tqdm:', tqdm.__version__)"
if %ERRORLEVEL% NEQ 0 echo [FAIL] tqdm not found

python -c "import spectral; print('[OK] spectral: OK')"
if %ERRORLEVEL% NEQ 0 echo [FAIL] spectral not found

echo.
python -c "import torch; print('[GPU]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

echo.
echo ==========================================
echo Verification complete.
echo ==========================================
