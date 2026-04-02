"""
INT8 dynamic quantization for CPU-optimized inference.
"""
import os, sys, logging
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelQuantizer:

    def __init__(self, model, output_dir="deployment/quantized"):
        self.model      = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def quantize_dynamic(self):
        """
        Dynamic INT8 quantization — no calibration data needed.
        Quantizes Linear layers for fast CPU inference.
        """
        self.model.eval().cpu()

        quantized = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        out_path = self.output_dir / "litenet_int8.pth"
        torch.save(quantized.state_dict(), out_path)
        logger.info(f"INT8 quantized model saved → {out_path}")

        # Size comparison
        orig_path = self.output_dir / "_temp_fp32.pth"
        torch.save(self.model.state_dict(), orig_path)
        orig_mb  = os.path.getsize(orig_path) / 1e6
        quant_mb = os.path.getsize(out_path)  / 1e6
        os.remove(orig_path)
        logger.info(f"FP32 size: {orig_mb:.2f} MB  →  INT8 size: {quant_mb:.2f} MB  "
                    f"(compression: {orig_mb/max(quant_mb,0.001):.1f}x)")

        # Latency comparison
        import time
        dummy_s = torch.randn(128, 204)
        dummy_p = torch.randn(128, 3, 3, 3)

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(50): self.model(dummy_s, dummy_p)
            fp32_ms = (time.perf_counter() - t0) / 50 * 1000

            t0 = time.perf_counter()
            for _ in range(50): quantized(dummy_s, dummy_p)
            int8_ms = (time.perf_counter() - t0) / 50 * 1000

        logger.info(f"FP32 latency: {fp32_ms:.2f}ms  |  INT8 latency: {int8_ms:.2f}ms  "
                    f"(speedup: {fp32_ms/max(int8_ms,0.001):.1f}x)")

        return quantized

    def save_torchscript(self):
        """TorchScript export for language-agnostic deployment."""
        self.model.eval().cpu()
        dummy_s = torch.randn(1, 204)
        dummy_p = torch.randn(1, 3, 3, 3)
        scripted = torch.jit.trace(self.model, (dummy_s, dummy_p))
        out_path = self.output_dir / "litenet_scripted.pt"
        scripted.save(str(out_path))
        logger.info(f"TorchScript model saved → {out_path}")
        return str(out_path)
