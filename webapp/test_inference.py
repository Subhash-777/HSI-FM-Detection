"""
Standalone diagnostic test for HSIDetect inference pipeline.
Run: python webapp/test_inference.py
Tests every step with timing, prints exactly where it hangs or fails.
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("HSIDetect Inference Diagnostic Test")
print("=" * 60)

# ── Step 1: Import check ──────────────────────────────────────
print("\n[1/7] Checking imports...")
t = time.time()
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except Exception as e:
    print(f"  ✗ numpy FAILED: {e}"); sys.exit(1)

try:
    from PIL import Image
    print(f"  ✓ Pillow OK")
except Exception as e:
    print(f"  ✗ Pillow FAILED: {e}"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    print(f"  ✓ matplotlib OK")
except Exception as e:
    print(f"  ✗ matplotlib FAILED: {e}"); sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter, binary_erosion
    print(f"  ✓ scipy OK")
except Exception as e:
    print(f"  ✗ scipy FAILED: {e}"); sys.exit(1)

try:
    import torch
    print(f"  ✓ torch {torch.__version__} | CUDA={torch.cuda.is_available()}")
except Exception as e:
    print(f"  ✗ torch FAILED: {e}"); sys.exit(1)

print(f"  → imports OK ({time.time()-t:.2f}s)")

# ── Step 2: Import AnomalyPredictor ──────────────────────────
print("\n[2/7] Importing AnomalyPredictor from inference.py...")
t = time.time()
try:
    # Add webapp dir to path
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, webapp_dir)
    from inference import AnomalyPredictor
    print(f"  ✓ AnomalyPredictor imported ({time.time()-t:.2f}s)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 3: Instantiate predictor ────────────────────────────
print("\n[3/7] Creating AnomalyPredictor instance (prototype mode)...")
t = time.time()
try:
    predictor = AnomalyPredictor(model=None, device=None, threshold=0.5)
    print(f"  ✓ Predictor created ({time.time()-t:.2f}s)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 4: Create synthetic test image ──────────────────────
print("\n[4/7] Creating synthetic test image (wheat + foreign object)...")
t = time.time()
try:
    # 400x400 golden wheat background with a dark brown stone patch
    img_np = np.ones((400, 400, 3), dtype=np.uint8)
    img_np[:, :, 0] = 210   # R — golden
    img_np[:, :, 1] = 160   # G
    img_np[:, :, 2] = 80    # B

    # White background border (simulates tray)
    img_np[:50, :]  = [255, 255, 255]
    img_np[-50:, :] = [255, 255, 255]
    img_np[:, :50]  = [255, 255, 255]
    img_np[:, -50:] = [255, 255, 255]

    # Dark brown stone (foreign object 1)
    img_np[180:210, 185:215] = [80, 50, 30]

    # Green plant debris (foreign object 2)
    img_np[220:230, 230:250] = [40, 120, 40]

    synthetic_img = Image.fromarray(img_np)
    print(f"  ✓ Synthetic image created: {synthetic_img.size} ({time.time()-t:.2f}s)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 5: Test background mask ─────────────────────────────
print("\n[5/7] Testing background mask...")
t = time.time()
try:
    bg_mask = predictor._background_mask(img_np)
    bg_pct  = bg_mask.mean() * 100
    grain_pct = (1 - bg_mask.mean()) * 100
    print(f"  ✓ Background mask OK ({time.time()-t:.2f}s)")
    print(f"     Background pixels : {bg_pct:.1f}%")
    print(f"     Grain pixels      : {grain_pct:.1f}%")
    if grain_pct < 5:
        print(f"  ⚠ WARNING: Very little grain detected — BG_BRIGHTNESS may be too low")
    if grain_pct > 95:
        print(f"  ⚠ WARNING: Almost no background detected — check BG_BRIGHTNESS={predictor.BG_BRIGHTNESS}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 6: Test score map ────────────────────────────────────
print("\n[6/7] Testing prototype score map...")
t = time.time()
try:
    score_map, bg_mask_out, z_smooth = predictor._prototype_score_map(img_np)
    grain_mask = ~bg_mask_out
    if grain_mask.any():
        grain_scores = score_map[grain_mask]
        print(f"  ✓ Score map OK ({time.time()-t:.2f}s)")
        print(f"     Score map shape   : {score_map.shape}")
        print(f"     Grain score min   : {grain_scores.min():.4f}")
        print(f"     Grain score max   : {grain_scores.max():.4f}")
        print(f"     Grain score mean  : {grain_scores.mean():.4f}")
        print(f"     Grain score std   : {grain_scores.std():.4f}")
        if grain_scores.std() < 0.01:
            print(f"  ⚠ WARNING: Score std very low — all pixels scoring similarly")
        else:
            print(f"  ✓ Good score variance — anomalies should be detectable")
    else:
        print(f"  ⚠ No grain pixels found in score map")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 7: Full predict() call ───────────────────────────────
print("\n[7/7] Running full predict() on synthetic image...")
t = time.time()
try:
    result = predictor.predict(synthetic_img)
    elapsed = time.time() - t
    print(f"  ✓ predict() completed in {elapsed:.2f}s")
    print(f"\n  ── Results ──────────────────────────────")
    print(f"     Verdict          : {result['verdict']}")
    print(f"     is_anomaly       : {result['is_anomaly']}")
    print(f"     anomaly_pct      : {result['anomaly_pct']}%")
    print(f"     max_conf         : {result['max_conf']}%")
    print(f"     mean_conf        : {result['mean_conf']}%")
    print(f"     threshold        : {result['threshold']}")
    print(f"     sensitivity      : {result['sensitivity']}")
    print(f"     original b64     : {'OK' if result['original'].startswith('data:') else 'MISSING'}")
    print(f"     heatmap b64      : {'OK' if result['heatmap'].startswith('data:') else 'MISSING'}")
    print(f"     overlay b64      : {'OK' if result['overlay'].startswith('data:') else 'MISSING'}")

    # Verdict check
    if result['is_anomaly']:
        print(f"\n  ✓ PASS — Anomaly correctly detected on synthetic image")
    else:
        print(f"\n  ⚠ FAIL — Anomaly NOT detected (coverage={result['anomaly_pct']}%)")
        print(f"     → ZSCORE_THRESH={predictor.ZSCORE_THRESH} may be too high")
        print(f"     → MIN_ANOM_FRAC={predictor.MIN_ANOM_FRAC} may be too high")
        print(f"     → Try lowering ZSCORE_THRESH to 2.0 in inference.py")

except Exception as e:
    elapsed = time.time() - t
    print(f"  ✗ FAILED after {elapsed:.2f}s: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 8: Test with real image if exists ────────────────────
print("\n[BONUS] Looking for real image to test...")
test_image_paths = [
    os.path.join(os.path.dirname(__file__), "test_image.jpg"),
    os.path.join(os.path.dirname(__file__), "static", "test.jpg"),
    "test_image.jpg",
]
real_img_path = None
for p in test_image_paths:
    if os.path.exists(p):
        real_img_path = p
        break

if real_img_path:
    print(f"  Found: {real_img_path}")
    t = time.time()
    try:
        real_img = Image.open(real_img_path).convert("RGB")
        print(f"  Image size: {real_img.size}")
        result2 = predictor.predict(real_img)
        print(f"  ✓ Real image predict() OK in {time.time()-t:.2f}s")
        print(f"     Verdict: {result2['verdict']} | coverage={result2['anomaly_pct']}%")
    except Exception as e:
        print(f"  ✗ Real image FAILED: {e}")
        traceback.print_exc()
else:
    print("  No real image found — skipping")
    print("  TIP: Copy your wheat image to webapp/test_image.jpg to test")

# ── Flask API test ────────────────────────────────────────────
print("\n[API] Testing Flask /predict endpoint (server must be running)...")
try:
    import requests
    import io as _io

    buf = _io.BytesIO()
    synthetic_img.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✓ API responded OK")
        print(f"     Verdict: {data.get('verdict')} | coverage={data.get('anomaly_pct')}%")
        if "error" in data:
            print(f"  ✗ API returned error: {data['error']}")
    else:
        print(f"  ✗ API returned HTTP {resp.status_code}: {resp.text[:200]}")
except ImportError:
    print("  ⚠ requests not installed — skipping API test")
    print("    Install with: pip install requests")
except Exception as e:
    print(f"  ⚠ API test skipped (server not running or error): {e}")

print("\n" + "=" * 60)
print("Diagnostic complete.")
print("=" * 60)
