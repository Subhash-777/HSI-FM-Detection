"""
HSIDetect Inference — Fixed Prototype + LiteNet-ready
Prototype mode: background-masked, grain-only color z-score anomaly detection.
LiteNet mode: full spectral+spatial pixel classifier (when model is loaded).
"""
import io, base64, logging
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, binary_erosion


logger = logging.getLogger(__name__)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class AnomalyPredictor:
    """
    Dual-mode predictor.
    Even when LiteNet is loaded, if RGB-derived pseudo-spectra produce
    degenerate (uniform ~0.5) outputs, automatically falls back to
    the prototype color z-score method which works reliably on RGB.
    """

    # ── Thresholds ────────────────────────────────────────────────────
    ZSCORE_THRESH    = 3.2   # verdict threshold
    OVERLAY_Z_MIN    = 3.8   # ← NEW: stricter draw threshold for overlay/heatmap
    MIN_ANOM_FRAC    = 0.8   # % grain area to trigger ANOMALY banner

    # ── Background / smoothing ────────────────────────────────────────
    BG_BRIGHTNESS    = 200
    GAUSS_SCORE      = 0.7   # ← was 1.0 → tighter blobs, less bloom
    GAUSS_ALPHA      = 1.5   # ← was 3.0 → overlay hugs the object

    # ── Inference ─────────────────────────────────────────────────────
    MAX_DIM          = 256
    BATCH_SIZE       = 16384
    DEGEN_STD_THRESH = 0.05


    def __init__(self, model=None, device=None, threshold: float = 0.5):
        self.model     = model
        self.device    = device
        self.threshold = threshold
        self.use_model = model is not None
        mode = "LiteNet" if self.use_model else "Prototype (color z-score)"
        logger.info(f"AnomalyPredictor ready | mode={mode}")


    # ═══════════════════════════════════════════════════════════════════
    #  SHARED UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _background_mask(self, img_np: np.ndarray) -> np.ndarray:
        rgb_mean     = img_np.mean(axis=2)
        bg_raw       = rgb_mean > self.BG_BRIGHTNESS
        grain_eroded = binary_erosion(~bg_raw, iterations=2)
        return ~grain_eroded


    def _make_heatmap(self, z_full: np.ndarray,
                      bg_full: np.ndarray) -> Image.Image:
        """
        FIX: normalize relative to grain pixels only so only true
        outliers (z >> OVERLAY_Z_MIN) appear bright; grain background
        stays near-black instead of saturated red.
        """
        grain_full   = ~bg_full
        heatmap_data = np.zeros_like(z_full, dtype=np.float64)

        if grain_full.any():
            grain_z      = z_full[grain_full]
            # Clip to [OVERLAY_Z_MIN, 99th-percentile] then normalize
            z_low        = self.OVERLAY_Z_MIN
            z_high       = max(np.percentile(grain_z, 99.5), z_low + 1e-6)
            normalized   = np.clip((z_full - z_low) / (z_high - z_low), 0.0, 1.0)
            normalized[bg_full] = 0.0
            heatmap_data = normalized

        heatmap_rgba = cm.inferno(heatmap_data)          # ← inferno: black→yellow
        heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_rgb[bg_full] = [0, 0, 0]
        return Image.fromarray(heatmap_rgb)


    def _make_overlay(self, orig_np: np.ndarray,
                      z_full: np.ndarray,
                      bg_full: np.ndarray) -> Image.Image:
        """
        FIX: use OVERLAY_Z_MIN (stricter than verdict threshold) so
        the red mask only covers pixels with truly high z-scores.
        """
        # Pixels must exceed OVERLAY_Z_MIN to get any red at all
        excess       = np.clip(z_full - self.OVERLAY_Z_MIN, 0, None)
        spread       = max(self.OVERLAY_Z_MIN * 0.5, 1e-6)
        alpha_raw    = np.clip(excess / spread, 0, 1).astype(np.float64)
        alpha_raw[bg_full] = 0.0

        alpha_sm     = gaussian_filter(alpha_raw, sigma=self.GAUSS_ALPHA)
        alpha_sm[bg_full] = 0.0
        alpha        = np.clip(alpha_sm, 0, 0.88)[:, :, np.newaxis]

        red          = np.array([255, 30, 30], dtype=np.float32)
        overlay      = (1 - alpha) * orig_np.astype(np.float32) + alpha * red
        return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


    def _build_result(self, score_full, bg_full, orig_rgb, z_full=None):
        grain_full  = ~bg_full
        grain_total = grain_full.sum()

        if z_full is None:
            grain_scores = (score_full[grain_full]
                            if grain_full.any() else score_full.flatten())
            mu    = grain_scores.mean()
            sigma = grain_scores.std() + 1e-8
            z_full = np.zeros_like(score_full)
            if grain_full.any():
                z_full[grain_full] = (score_full[grain_full] - mu) / sigma

        # Verdict uses ZSCORE_THRESH
        anom_mask  = (z_full >= self.ZSCORE_THRESH) & grain_full
        anom_pct   = float(anom_mask.sum() / max(grain_total, 1) * 100)
        is_anomaly = anom_pct >= self.MIN_ANOM_FRAC

        max_conf  = round(float(score_full[anom_mask].max()  * 100)
                          if anom_mask.any() else
                          float(score_full[grain_full].max() * 100)
                          if grain_full.any() else 0.0, 1)
        mean_conf = round(float(score_full[grain_full].mean() * 100)
                          if grain_full.any() else 0.0, 1)

        orig_np     = np.array(orig_rgb).astype(np.float32)
        # FIX: pass z_full (not score_full) into heatmap for correct normalization
        heatmap_img = self._make_heatmap(z_full, bg_full)
        overlay_img = self._make_overlay(orig_np, z_full, bg_full)

        verdict = "ANOMALY DETECTED" if is_anomaly else "SAFE"
        logger.info(f"{verdict} | coverage={anom_pct:.2f}% "
                    f"| max={max_conf}% | mean={mean_conf}%")

        return {
            "verdict":     verdict,
            "is_anomaly":  is_anomaly,
            "anomaly_pct": round(anom_pct, 2),
            "max_conf":    max_conf,
            "mean_conf":   mean_conf,
            "threshold":   round(float(self.ZSCORE_THRESH), 2),
            "sensitivity": f"z > {self.ZSCORE_THRESH}",
            "original":    pil_to_base64(orig_rgb),
            "heatmap":     pil_to_base64(heatmap_img),
            "overlay":     pil_to_base64(overlay_img),
        }


    # ═══════════════════════════════════════════════════════════════════
    #  PROTOTYPE MODE
    # ═══════════════════════════════════════════════════════════════════

    def _prototype_score_map(self, img_np: np.ndarray):
        H, W       = img_np.shape[:2]
        bg_mask    = self._background_mask(img_np)
        grain_mask = ~bg_mask

        if grain_mask.sum() < 100:
            grain_mask = np.ones((H, W), dtype=bool)
            bg_mask    = np.zeros((H, W), dtype=bool)

        ycbcr = np.array(
            Image.fromarray(img_np).convert("YCbCr")
        ).astype(np.float32)
        rgb_f = img_np.astype(np.float32)

        med_ycbcr = np.median(ycbcr[grain_mask], axis=0)
        std_ycbcr = np.std(ycbcr[grain_mask],   axis=0) + 1e-6
        med_rgb   = np.median(rgb_f[grain_mask], axis=0)
        std_rgb   = np.std(rgb_f[grain_mask],    axis=0) + 1e-6

        z_ycbcr = np.abs((ycbcr - med_ycbcr) / std_ycbcr)
        z_rgb   = np.abs((rgb_f - med_rgb)   / std_rgb)

        z_Y  = z_ycbcr[:, :, 0]
        z_Cb = z_ycbcr[:, :, 1]
        z_Cr = z_ycbcr[:, :, 2]
        z_G  = z_rgb[:, :, 1]

        z_combined = (0.25 * z_Y  + 0.30 * z_Cb +
                      0.25 * z_Cr + 0.20 * z_G)
        z_combined[bg_mask] = 0.0

        # FIX: tighter sigma=0.7 (was 1.0) → less spatial bloom
        z_smooth = gaussian_filter(z_combined.astype(np.float64),
                                   sigma=self.GAUSS_SCORE)
        z_smooth[bg_mask] = 0.0

        # score_map: min-max normalized over grain pixels only
        score_map = np.zeros((H, W), dtype=np.float32)
        if grain_mask.any():
            g_vals = z_smooth[grain_mask]
            g_min, g_max = g_vals.min(), g_vals.max() + 1e-6
            score_map[grain_mask] = (
                (z_smooth[grain_mask] - g_min) / (g_max - g_min)
            ).astype(np.float32)

        return score_map, bg_mask, z_smooth


    def _predict_prototype(self, orig_rgb: Image.Image) -> dict:
        orig_w, orig_h = orig_rgb.size

        PROTO_DIM = 256
        scale  = min(PROTO_DIM / max(orig_w, orig_h), 1.0)
        inf_w  = max(int(orig_w * scale), 8)
        inf_h  = max(int(orig_h * scale), 8)
        img_np = np.array(orig_rgb.resize((inf_w, inf_h), Image.LANCZOS))

        score_map, bg_mask, z_smooth = self._prototype_score_map(img_np)

        # Upscale score and z maps to original resolution
        score_full = np.array(
            Image.fromarray((score_map * 255).astype(np.uint8))
                 .resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        # FIX: upscale z_smooth directly (not just score) for accurate thresholding
        z_full_small = np.clip(z_smooth, 0, None).astype(np.float32)
        z_max        = z_full_small.max() + 1e-6
        z_full = np.array(
            Image.fromarray((z_full_small / z_max * 255).astype(np.uint8))
                 .resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0 * z_max

        bg_full = np.array(
            Image.fromarray(bg_mask.astype(np.uint8) * 255)
                 .resize((orig_w, orig_h), Image.NEAREST)
        ) > 127

        z_full[bg_full]     = 0.0
        score_full[bg_full] = 0.0

        return self._build_result(score_full, bg_full, orig_rgb, z_full)


    # ═══════════════════════════════════════════════════════════════════
    #  LITENET MODE  (with auto-fallback if scores are degenerate)
    # ═══════════════════════════════════════════════════════════════════

    N_BANDS    = 204
    SPEC_BASIS = None

    @classmethod
    def _get_spectral_basis(cls):
        if cls.SPEC_BASIS is None:
            wl      = np.linspace(400, 1000, cls.N_BANDS, dtype=np.float32)
            centres = np.array([450., 550., 650.], dtype=np.float32)
            sigma   = 40.0
            cls.SPEC_BASIS = np.stack([
                np.exp(-((wl - c) ** 2) / (2 * sigma ** 2))
                for c in centres
            ], axis=0).astype(np.float32)
        return cls.SPEC_BASIS

    def _rgb_to_spectra(self, rgb_pixels: np.ndarray) -> np.ndarray:
        basis   = self._get_spectral_basis()
        spectra = rgb_pixels @ basis
        s_min   = spectra.min(1, keepdims=True)
        s_max   = spectra.max(1, keepdims=True) + 1e-8
        spectra = (spectra - s_min) / (s_max - s_min)
        b_mean  = spectra.mean(0, keepdims=True)
        b_std   = spectra.std(0,  keepdims=True) + 1e-8
        return np.clip((spectra - b_mean) / b_std, -3., 3.).astype(np.float32)

    def _extract_inputs(self, img_np: np.ndarray):
        H, W   = img_np.shape[:2]
        img_f  = img_np.astype(np.float32) / 255.0
        padded = np.pad(img_f, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        N      = H * W
        spectra = self._rgb_to_spectra(img_f.reshape(N, 3))
        patches = np.stack([
            padded[r:r+H, c:c+W, :].reshape(N, 3)
            for r in range(3) for c in range(3)
        ], axis=1).reshape(N, 3, 3, 3).transpose(0, 3, 1, 2).astype(np.float32)
        return spectra, patches

    @torch.no_grad()
    def _run_litenet(self, spectra: np.ndarray,
                     spatial: np.ndarray) -> np.ndarray:
        probs   = []
        use_amp = self.device.type == "cuda"
        for i in range(0, len(spectra), self.BATCH_SIZE):
            s = torch.from_numpy(spectra[i:i+self.BATCH_SIZE]).to(self.device)
            p = torch.from_numpy(spatial[i:i+self.BATCH_SIZE]).to(self.device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(s, p)
            else:
                logits = self.model(s, p)
            probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        return np.concatenate(probs)

    def _predict_litenet(self, orig_rgb: Image.Image) -> dict:
        orig_w, orig_h = orig_rgb.size
        scale  = min(self.MAX_DIM / max(orig_w, orig_h), 1.0)
        inf_w  = max(int(orig_w * scale), 8)
        inf_h  = max(int(orig_h * scale), 8)
        img_np = np.array(orig_rgb.resize((inf_w, inf_h), Image.LANCZOS))

        # ── Fast degeneracy pre-check on 64 random pixels ─────────────
        H, W   = img_np.shape[:2]
        rng    = np.random.default_rng(42)
        idx_r  = rng.integers(0, H, 64)
        idx_c  = rng.integers(0, W, 64)
        sample_rgb  = (img_np[idx_r, idx_c].astype(np.float32) / 255.0)
        sample_spec = self._rgb_to_spectra(sample_rgb)
        sample_spat = np.zeros((64, 3, 3, 3), dtype=np.float32)

        with torch.no_grad():
            s = torch.from_numpy(sample_spec).to(self.device)
            p = torch.from_numpy(sample_spat).to(self.device)
            logits = self.model(s, p)
            probe  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        if probe.std() < self.DEGEN_STD_THRESH:
            logger.warning(
                f"LiteNet degenerate on probe (std={probe.std():.4f}) "
                f"— using prototype mode"
            )
            return self._predict_prototype(orig_rgb)

        # ── Full LiteNet inference ─────────────────────────────────────
        spectra, spatial = self._extract_inputs(img_np)
        raw_scores = self._run_litenet(spectra, spatial).reshape(inf_h, inf_w)

        sm        = gaussian_filter(raw_scores.astype(np.float64), sigma=1.0)
        z         = (sm - sm.mean()) / (sm.std() + 1e-8)
        score_map = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

        score_full = np.array(
            Image.fromarray((score_map * 255).astype(np.uint8))
                 .resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        z_max      = max(z.max(), 1e-6)
        z_full_up  = np.array(
            Image.fromarray(
                np.clip(z / z_max * 255, 0, 255).astype(np.uint8)
            ).resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0 * z_max

        bg_mask = self._background_mask(np.array(orig_rgb))
        bg_full = np.array(
            Image.fromarray(bg_mask.astype(np.uint8) * 255)
                 .resize((orig_w, orig_h), Image.NEAREST)
        ) > 127
        score_full[bg_full] = 0.0
        z_full_up[bg_full]  = 0.0

        return self._build_result(score_full, bg_full, orig_rgb, z_full_up)


    # ═══════════════════════════════════════════════════════════════════
    #  PUBLIC ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════

    def predict(self, pil_image: Image.Image) -> dict:
        orig_rgb = pil_image.convert("RGB")
        if self.use_model:
            return self._predict_litenet(orig_rgb)
        return self._predict_prototype(orig_rgb)
