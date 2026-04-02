.PHONY: help setup-conda verify extract preprocess signatures synthetic train evaluate export clean test clean-env clean-preprocessed

CONDA_ENV := hsi-fm

help:
	@echo HSI-FM-Detection Project Commands (Windows)
	@echo ==========================================
	@echo setup-conda   - Setup conda environment
	@echo verify        - Verify installation
	@echo extract       - Extract all datasets
	@echo preprocess    - Preprocess and harmonize data
	@echo signatures    - Extract FM spectral signatures
	@echo synthetic     - Generate synthetic dataset
	@echo train         - Train model (both phases)
	@echo evaluate      - Evaluate trained model
	@echo export        - Export model for deployment
	@echo pipeline      - Run full pipeline
	@echo test          - Run quick test
	@echo clean         - Clean generated data
	@echo clean-env     - Remove conda environment

setup-conda:
	@echo Setting up conda environment...
	@call scripts\setup_conda_env.bat

verify:
	@echo Verifying installation...
	@call scripts\verify_installation.bat

extract:
	@echo Extracting datasets...
	@call conda activate $(CONDA_ENV) && python -m src.data_extraction.extract_hsifood
	@call conda activate $(CONDA_ENV) && python -m src.data_extraction.extract_agrifood

preprocess:
	@echo Running preprocessing...
	@call scripts\run_preprocessing.bat

signatures:
	@echo Extracting FM signatures...
	@call conda activate $(CONDA_ENV) && python -m src.signature_extraction.fm_signature_extractor
	@call conda activate $(CONDA_ENV) && python -m src.signature_extraction.visualize_signatures

synthetic:
	@echo Generating synthetic data...
	@call scripts\run_synthetic_gen.bat

train:
	@echo Training model...
	@call scripts\run_training.bat > training_log.txt 2>&1

evaluate:
	@echo Evaluating model...
	@call scripts\run_evaluation.bat

export:
	@echo Exporting model...
	@call scripts\export_model.bat

pipeline:
	@echo Running full pipeline...
	@call scripts\run_full_pipeline.bat

test:
	@echo Running quick test...
	@call scripts\quick_test.bat

clean-preprocessed:
	@echo Removing preprocessed data...
	@del /Q data\synthetic\train_pt\*.pt 2>nul
	@del /Q data\synthetic\val_pt\*.pt 2>nul
	@del /Q data\processed\harmonized_204bands\*_preprocessed.pt 2>nul
	@rmdir /S /Q experiments\phase1_synthetic 2>nul
	@rmdir /S /Q experiments\phase2_real 2>nul
	@echo Done

clean:
	@call scripts\clean_data.bat

clean-env:
	@echo Removing conda environment...
	@conda env remove -n $(CONDA_ENV) -y
	@echo Environment removed!