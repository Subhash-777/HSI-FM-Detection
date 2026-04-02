@echo off
REM HSI Model Training Pipeline - Phase 1 + Phase 2

echo Training model...
echo ==========================================
echo HSI Model Training Pipeline
echo ==========================================

call conda activate hsi-fm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment hsi-fm
    exit /b 1
)

REM ---- Preprocessing status check ----
echo.
if exist "data\synthetic\train_pt\synthetic_train_preprocessed.pt" (
    echo [OK] Synthetic .pt files exist, skipping synthetic preprocessing
) else (
    echo [INFO] Synthetic .pt files missing, running preprocessing...
    call scripts\run_preprocessing.bat
    if %ERRORLEVEL% NEQ 0 exit /b 1
)

if exist "data\processed\harmonized_204bands\agrifood_train_preprocessed.pt" (
    echo [OK] Real .pt files exist, skipping real preprocessing
) else (
    echo [INFO] Real .pt files missing, running preprocessing...
    call scripts\run_preprocessing.bat
    if %ERRORLEVEL% NEQ 0 exit /b 1
)

echo.
python scripts\check_preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Preprocessing check failed
    exit /b 1
)

echo [OK] All files OK. Ready for training.

REM ---- Phase 1: Synthetic Pre-training ----
echo.
echo ==========================================
echo Phase 1: Synthetic Pre-training
echo ==========================================

if exist "experiments\phase1_synthetic\best_model.pth" (
    echo [OK] Phase1 model exists, skipping training
    goto :phase2
)

python scripts\train_phase1.py
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Phase 1 training failed
    exit /b 1
)
echo [OK] Phase 1 complete!

:phase2
REM ---- Phase 2: Real Data Fine-tuning ----
echo.
echo ==========================================
echo Phase 2: Real Data Fine-tuning
echo ==========================================

if exist "experiments\phase2_real\best_model.pth" (
    echo [OK] Phase2 model exists, skipping fine-tuning
    goto :done
)

python scripts\train_phase2.py
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Phase 2 fine-tuning failed
    exit /b 1
)
echo [OK] Phase 2 complete!

:done
echo.
echo ==========================================
echo [OK] TRAINING PIPELINE COMPLETE!
echo ==========================================
