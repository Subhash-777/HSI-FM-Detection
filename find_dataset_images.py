# run: python generate_food_tests.py
from PIL import Image
import numpy as np, os

OUT = r"E:\SE\webapp\test_images"
os.makedirs(OUT, exist_ok=True)

def make_tray(H, W, grain_rgb, noise=12):
    img = np.ones((H, W, 3), dtype=np.float32)
    img[:, :] = grain_rgb
    img += np.random.randn(H, W, 3) * noise
    # White tray border
    img[:40, :] = 252; img[-40:, :] = 252
    img[:, :40] = 252; img[:, -40:] = 252
    return np.clip(img, 0, 255).astype(np.uint8)

def save(img, name):
    path = os.path.join(OUT, name)
    Image.fromarray(img).save(path)
    print(f"  ✓ {name}")

# ── 1. Wheat (Blé) — CLEAN ───────────────────────────────────
img = make_tray(600, 800, [210, 175, 95])
save(img, "01_wheat_CLEAN.jpg")

# ── 2. Wheat — Stone + Rubber ────────────────────────────────
img = make_tray(600, 800, [210, 175, 95])
img[280:310, 370:400] = [95, 80, 68]   # brown stone
img[400:415, 500:540] = [28, 28, 32]   # black rubber
save(img, "02_wheat_STONE_RUBBER.jpg")

# ── 3. Barley (Orge) — CLEAN ─────────────────────────────────
img = make_tray(600, 800, [195, 160, 80])
save(img, "03_barley_CLEAN.jpg")

# ── 4. Barley — Green plant debris ───────────────────────────
img = make_tray(600, 800, [195, 160, 80])
img[200:215, 300:420] = [65, 128, 52]  # green stem
img[350:370, 550:575] = [55, 110, 44]  # green leaf fragment
save(img, "04_barley_GREEN_DEBRIS.jpg")

# ── 5. Oat (Avoine) — CLEAN ──────────────────────────────────
img = make_tray(600, 800, [220, 195, 130])
save(img, "05_oat_CLEAN.jpg")

# ── 6. Oat — Multiple contaminants ───────────────────────────
img = make_tray(600, 800, [220, 195, 130])
img[150:175, 200:235] = [88, 72, 60]   # stone
img[300:315, 450:490] = [60, 125, 48]  # green debris
img[450:465, 300:320] = [22, 22, 26]   # black plastic
save(img, "06_oat_MULTI_CONTAMINANT.jpg")

# ── 7. Rice (Riz) — CLEAN ────────────────────────────────────
img = make_tray(600, 800, [240, 232, 215], noise=6)
save(img, "07_rice_CLEAN.jpg")

# ── 8. Rice — Dark contaminant (husk/stone) ──────────────────
img = make_tray(600, 800, [240, 232, 215], noise=6)
img[270:295, 380:410] = [110, 90, 70]  # brown husk
img[380:395, 520:550] = [180, 50, 50]  # red foreign grain
save(img, "08_rice_HUSK_REDGRAIN.jpg")

print(f"\nAll 8 images saved to: {OUT}")
print("Upload them to http://localhost:5000 to test!")
