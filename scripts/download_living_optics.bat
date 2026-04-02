@echo off
REM Download LivingOptics / dataset files

echo ==========================================
echo Download Datasets
echo ==========================================

call conda activate hsi-fm

echo [INFO] Checking data directories...
if not exist "data\raw\HSI-AgriFoodAnomaly" mkdir "data\raw\HSI-AgriFoodAnomaly"
if not exist "data\raw\HSIFoodIngr-64" mkdir "data\raw\HSIFoodIngr-64"

echo.
echo NOTE: Datasets require manual download due to license restrictions.
echo.
echo 1. HSI-AgriFoodAnomaly:
echo    Download from your data source and place in:
echo    data\raw\HSI-AgriFoodAnomaly\extracted\
echo.
echo 2. HSIFoodIngr-64:
echo    Download from your data source and place in:
echo    data\raw\HSIFoodIngr-64\extracted\
echo.

REM If you have a download script, call it here
REM python src\data\download.py --dataset agrifood --output data\raw\HSI-AgriFoodAnomaly
REM python src\data\download.py --dataset hsifood --output data\raw\HSIFoodIngr-64

echo [INFO] Verifying downloaded files...
python -c "
from pathlib import Path
agrifood = Path('data/raw/HSI-AgriFoodAnomaly/extracted')
hsifood = Path('data/raw/HSIFoodIngr-64/extracted')
print(f'HSI-AgriFoodAnomaly: {\"FOUND\" if agrifood.exists() else \"MISSING - please download\"}')
print(f'HSIFoodIngr-64:      {\"FOUND\" if hsifood.exists() else \"MISSING - please download\"}')
"
echo.
echo ==========================================
