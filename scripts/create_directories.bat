@echo off
REM Create all project directories

echo ==========================================
echo Creating Project Directory Structure
echo ==========================================

mkdir data\raw 2>nul
mkdir data\raw\HSI-AgriFoodAnomaly\extracted 2>nul
mkdir data\raw\HSIFoodIngr-64\extracted 2>nul
mkdir data\synthetic\train_pt 2>nul
mkdir data\synthetic\val_pt 2>nul
mkdir data\processed\harmonized_204bands 2>nul
mkdir data\processed\spectral_signatures 2>nul
mkdir data\processed\metadata 2>nul
mkdir data\processed\pca_models 2>nul
mkdir experiments\phase1_synthetic 2>nul
mkdir experiments\phase2_real 2>nul
mkdir experiments\ensemble 2>nul
mkdir outputs\models 2>nul
mkdir outputs\figures 2>nul
mkdir outputs\reports 2>nul
mkdir logs 2>nul
mkdir results 2>nul

echo.
echo [OK] All directories created successfully!
echo.
