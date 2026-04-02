@echo off
REM Generate synthetic training data

echo ==========================================
echo Synthetic Data Generation
echo ==========================================

call conda activate hsi-fm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)

echo [INFO] Starting synthetic data generation...
echo [INFO] Config: config\config.yaml
echo.

python -c "
from src.synthetic.generator import SyntheticGenerator
gen = SyntheticGenerator('config/config.yaml')
gen.generate()
"
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Synthetic generation failed
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Synthetic data generation complete!
echo ==========================================
