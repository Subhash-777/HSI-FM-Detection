@echo off
REM Run full preprocessing pipeline

echo ==========================================
echo HSI Preprocessing Pipeline
echo ==========================================

call conda activate hsi-fm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment hsi-fm
    exit /b 1
)

REM --- Synthetic preprocessing ---
echo.
echo [STEP 1] Checking synthetic .pt files...
if exist "data\synthetic\train_pt\synthetic_train_preprocessed.pt" (
    if exist "data\synthetic\val_pt\synthetic_val_preprocessed.pt" (
        echo [SKIP] Synthetic .pt files exist, skipping synthetic preprocessing
        goto :real_preprocessing
    )
)

echo [INFO] Running synthetic preprocessing...
python -c "from src.preprocessing.preprocess_synthetic import run_synthetic_preprocessing; run_synthetic_preprocessing('config/config.yaml')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Synthetic preprocessing failed
    exit /b 1
)
echo [OK] Synthetic preprocessing complete

:real_preprocessing
REM --- Real data preprocessing ---
echo.
echo [STEP 2] Checking real .pt files...
if exist "data\processed\harmonized_204bands\agrifood_train_preprocessed.pt" (
    if exist "data\processed\harmonized_204bands\agrifood_val_preprocessed.pt" (
        echo [SKIP] Real .pt files exist, skipping real preprocessing
        goto :status_check
    )
)

echo [INFO] Running real data preprocessing...
python -c "from src.preprocessing.preprocess_real import run_real_preprocessing; run_real_preprocessing('config/config.yaml')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Real data preprocessing failed
    exit /b 1
)
echo [OK] Real data preprocessing complete

:status_check
echo.
echo [STEP 3] Running preprocessing status check...
python scripts\check_preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Preprocessing check failed
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Preprocessing pipeline complete!
echo ==========================================
