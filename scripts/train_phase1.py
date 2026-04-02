import sys
sys.path.insert(0, ".")

if __name__ == "__main__":
    from src.training import SyntheticTrainer

    trainer = SyntheticTrainer("config/config.yaml")
    trainer.train(
        train_data="data/synthetic/train_pt/synthetic_train_preprocessed.pt",
        val_data="data/synthetic/val_pt/synthetic_val_preprocessed.pt",
        use_preprocessed=True
    )
