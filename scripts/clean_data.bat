@echo off
REM Clean preprocessed data and experiment files

echo ==========================================
echo Clean Data
echo ==========================================
echo.
echo WARNING: This will delete preprocessed files!
echo Press Ctrl+C to cancel, or
pause

set /p CHOICE="Type YES to confirm deletion: "
if /i NOT "%CHOICE%"=="YES" (
    echo Cancelled.
    exit /b 0
)

echo.
echo [INFO] Cleaning preprocessed synthetic data...
if exist "data\synthetic\train_pt" (
    del /q "data\synthetic\train_pt\*.pt" 2>nul
    echo [OK] Synthetic train .pt files removed
)
if exist "data\synthetic\val_pt" (
    del /q "data\synthetic\val_pt\*.pt" 2>nul
    echo [OK] Synthetic val .pt files removed
)

echo [INFO] Cleaning preprocessed real data...
if exist "data\processed\harmonized_204bands" (
    del /q "data\processed\harmonized_204bands\*.pt" 2>nul
    echo [OK] Real .pt files removed
)

echo [INFO] Cleaning experiment files...
set /p CLEAN_EXP="Delete experiment models too? (YES/no): "
if /i "%CLEAN_EXP%"=="YES" (
    if exist "experiments\phase1_synthetic" rmdir /s /q "experiments\phase1_synthetic"
    if exist "experiments\phase2_real" rmdir /s /q "experiments\phase2_real"
    echo [OK] Experiment models removed
)

echo [INFO] Cleaning logs...
if exist "logs" del /q "logs\*.log" 2>nul

echo.
echo [OK] Clean complete!
