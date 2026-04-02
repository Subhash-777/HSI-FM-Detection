"""
Export LiteNet to ONNX for production inference.
"""
import os, sys, logging
sys.path.insert(0, ".")

import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXExporter:

    def __init__(self, model, device, output_dir="deployment/onnx"):
        self.model      = model
        self.device     = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, threshold: float = 0.5,
               n_bands: int = 204, spatial_size: int = 3):
        """Export to ONNX with dynamic batch size."""
        self.model.eval()
        self.model.cpu()

        # Dummy inputs
        dummy_spectra = torch.randn(1, n_bands)
        dummy_spatial = torch.randn(1, 3, spatial_size, spatial_size)

        onnx_path = self.output_dir / "litenet.onnx"

        torch.onnx.export(
            self.model,
            (dummy_spectra, dummy_spatial),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["spectra", "spatial"],
            output_names=["logits"],
            dynamic_axes={
                "spectra": {0: "batch_size"},
                "spatial": {0: "batch_size"},
                "logits":  {0: "batch_size"},
            },
        )
        logger.info(f"ONNX model exported → {onnx_path}")

        # Verify
        try:
            import onnx
            model_onnx = onnx.load(str(onnx_path))
            onnx.checker.check_model(model_onnx)
            logger.info("ONNX model verification: PASSED")
        except ImportError:
            logger.warning("onnx package not installed — skipping verification")

        # Save threshold alongside
        meta_path = self.output_dir / "model_metadata.json"
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "threshold":    threshold,
                "n_bands":      n_bands,
                "spatial_size": spatial_size,
                "input_names":  ["spectra", "spatial"],
                "output_name":  "logits",
                "postprocess":  "sigmoid → threshold",
            }, f, indent=2)
        logger.info(f"Metadata saved → {meta_path}")

        # Accuracy check with onnxruntime
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=["CPUExecutionProvider"])
            out  = sess.run(None, {
                "spectra": dummy_spectra.numpy(),
                "spatial": dummy_spatial.numpy(),
            })
            pytorch_out = self.model(dummy_spectra, dummy_spatial).detach().numpy()
            max_diff = float(np.abs(out[0] - pytorch_out).max())
            logger.info(f"PyTorch vs ONNX max diff: {max_diff:.2e}")
            if max_diff < 1e-4:
                logger.info("Numerical parity: PASSED")
        except ImportError:
            logger.warning("onnxruntime not installed — skipping parity check")

        return str(onnx_path)
