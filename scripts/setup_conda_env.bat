@echo off
REM Setup conda environment for HSI-FM-Detection

echo ==========================================
echo HSI-FM-Detection Environment Setup
echo ==========================================

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    exit /b 1
)

echo [INFO] Creating conda environment: hsi-fm
call conda create -n hsi-fm python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create conda environment
    exit /b 1
)

echo [INFO] Activating environment...
call conda activate hsi-fm

echo [INFO] Installing PyTorch with CUDA...
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo [INFO] Installing dependencies...
pip install numpy scipy scikit-learn h5py tqdm pyyaml matplotlib seaborn spectral

echo.
echo ==========================================
echo [OK] Environment setup complete!
echo     Activate with: conda activate hsi-fm
echo ==========================================
