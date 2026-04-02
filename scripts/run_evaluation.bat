@echo off
REM Run full evaluation with threshold tuning and TTA

echo ==========================================
echo HSI Model Evaluation Pipeline
echo ==========================================

call conda activate hsi-fm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)

set MODEL_PATH=experiments\phase2_real\best_model.pth
set VAL_DATA=data\processed\harmonized_204bands\agrifood_val_preprocessed.pt
set TEST_DATA=data\processed\harmonized_204bands\agrifood_test_preprocessed.pt

if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found: %MODEL_PATH%
    echo Please run run_training.bat first.
    exit /b 1
)

REM --- Step 1: Threshold tuning on validation set ---
echo.
echo [STEP 1] Threshold tuning on validation set...
python scripts\evaluate_with_threshold_tuning.py ^
    --model "%MODEL_PATH%" ^
    --data "%VAL_DATA%" ^
    --config config\config.yaml ^
    --output results\threshold_tuning_val.json ^
    --optimize-for f1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Threshold tuning failed
    exit /b 1
)
echo [OK] Threshold tuning complete. Results: results\threshold_tuning_val.json

REM --- Step 2: TTA evaluation on validation set ---
echo.
echo [STEP 2] TTA evaluation on validation set...
python scripts\evaluate_with_tta.py ^
    --model "%MODEL_PATH%" ^
    --data "%VAL_DATA%" ^
    --config config\config.yaml ^
    --threshold 0.45 ^
    --n-augs 4 ^
    --output results\tta_val_evaluation.json
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] TTA evaluation failed
    exit /b 1
)
echo [OK] TTA val evaluation complete. Results: results\tta_val_evaluation.json

REM --- Step 3: Final test evaluation ---
if exist "%TEST_DATA%" (
    echo.
    echo [STEP 3] Final evaluation on test set...
    python scripts\evaluate_with_tta.py ^
        --model "%MODEL_PATH%" ^
        --data "%TEST_DATA%" ^
        --config config\config.yaml ^
        --threshold 0.45 ^
        --n-augs 4 ^
        --output results\test_evaluation_final.json
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Test evaluation failed
        exit /b 1
    )
    echo [OK] Test evaluation complete. Results: results\test_evaluation_final.json
) else (
    echo [SKIP] Test data not found, skipping test evaluation
)

echo.
echo ==========================================
echo [OK] Evaluation pipeline complete!
echo     Check results\ folder for outputs
echo ==========================================
