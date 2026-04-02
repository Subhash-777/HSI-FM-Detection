@echo off
REM Run the complete HSI pipeline end-to-end

echo ==========================================
echo HSI Full Pipeline
echo ==========================================
echo  1. Create directories
echo  2. Preprocessing
echo  3. Training
echo  4. Evaluation
echo ==========================================

call conda activate hsi-fm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment hsi-fm
    exit /b 1
)

echo.
echo [PHASE 0] Creating directories...
call scripts\create_directories.bat
if %ERRORLEVEL% NEQ 0 ( echo [FATAL] Directory creation failed & exit /b 1 )

echo.
echo [PHASE 1] Running preprocessing...
call scripts\run_preprocessing.bat
if %ERRORLEVEL% NEQ 0 ( echo [FATAL] Preprocessing failed & exit /b 1 )

echo.
echo [PHASE 2] Running training...
call scripts\run_training.bat
if %ERRORLEVEL% NEQ 0 ( echo [FATAL] Training failed & exit /b 1 )

echo.
echo [PHASE 3] Running evaluation...
call scripts\run_evaluation.bat
if %ERRORLEVEL% NEQ 0 ( echo [FATAL] Evaluation failed & exit /b 1 )

echo.
echo ==========================================
echo [OK] FULL PIPELINE COMPLETE!
echo ==========================================
