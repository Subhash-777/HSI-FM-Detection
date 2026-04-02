"""
Export trained model for deployment: ONNX + INT8 + TorchScript
Usage: python scripts/export_model.py
"""
import os, sys, logging, json, yaml
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
from src.models.litenet import LiteNet
from src.deployment.export_onnx import ONNXExporter
from src.deployment.quantize_int8 import ModelQuantizer

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")  # export on CPU

    model_cfg = config.get("model", {})
    model = LiteNet(
        n_bands=model_cfg.get("input_bands", 204),
        spatial_channels=model_cfg.get("spatial_branch", {}).get("input_channels", 3),
        spectral_output_dim=model_cfg.get("spectral_branch", {}).get("channels", [None, None, 128])[-1],
        spatial_output_dim=model_cfg.get("spatial_branch", {}).get("channels", [None, 64])[-1],
        fusion_hidden_dim=model_cfg.get("fusion", {}).get("hidden_dim", 64),
        dropout=0.0,
        spectral_architecture=model_cfg.get("spectral_branch", {}).get("architecture", "simple"),
        spatial_input_size=3,
    )

    # Load best model + optimal threshold from evaluation
    ckpt = torch.load("experiments/phase2_real/best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_results_path = "results/evaluation/evaluation_results.json"
    threshold = 0.5
    if os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            threshold = json.load(f)["optimal_threshold"]
    print(f"Using threshold: {threshold:.4f}")

    # ONNX export
    exporter = ONNXExporter(model, device, output_dir="deployment/onnx")
    exporter.export(threshold=threshold)

    # INT8 + TorchScript
    quantizer = ModelQuantizer(model, output_dir="deployment/quantized")
    quantizer.quantize_dynamic()
    quantizer.save_torchscript()

    print("\n=== Deployment artifacts ready ===")
    print("  deployment/onnx/litenet.onnx")
    print("  deployment/onnx/model_metadata.json")
    print("  deployment/quantized/litenet_int8.pth")
    print("  deployment/quantized/litenet_scripted.pt")
