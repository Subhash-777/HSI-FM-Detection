import sys
sys.path.insert(0, ".")

if __name__ == "__main__":
    from src.training import RealDataFinetuner

    finetuner = RealDataFinetuner(
        config_path="config/config.yaml",
        pretrained_model_path="experiments/phase1_synthetic/best_model.pth"
    )
    finetuner.finetune(
        train_data="data/processed/harmonized_204bands/agrifood_train_preprocessed.pt",
        val_data="data/processed/harmonized_204bands/agrifood_val_preprocessed.pt",
        use_preprocessed=True
    )
