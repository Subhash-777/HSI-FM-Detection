"""
Flask backend for HSI Anomaly Detection Web App.
Prototype mode: color z-score anomaly detection on RGB images.
"""
import os, sys, json, logging, datetime, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

from inference import AnomalyPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH    = os.path.join(BASE_DIR, "results", "evaluation", "evaluation_results.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Threshold ─────────────────────────────────────────────────────────────────
threshold = 0.5
if os.path.exists(EVAL_PATH):
    try:
        with open(EVAL_PATH) as f:
            threshold = json.load(f).get("optimal_threshold", 0.5)
        logger.info(f"Loaded threshold: {threshold:.4f}")
    except Exception:
        pass

# ── Predictor ─────────────────────────────────────────────────────────────────
model_mode  = "prototype"
model_label = "Prototype (Color Z-Score)"

predictor = AnomalyPredictor(model=None, device=None, threshold=threshold)
logger.info(f"HSIDetect ready | mode={model_mode} | threshold={threshold:.4f}")


# ── History helpers ───────────────────────────────────────────────────────────
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entry: dict):
    history = load_history()
    history.insert(0, entry)
    history = history[:100]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template(
        "about.html",
        threshold=round(threshold * 100, 1),
        model_mode=model_mode,
        model_label=model_label
    )


@app.route("/history")
def history():
    return render_template("history.html", entries=load_history())


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result    = predictor.predict(pil_image)

        save_history({
            "id":          str(uuid.uuid4())[:8],
            "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict":     result["verdict"],
            "anomaly_pct": result["anomaly_pct"],
            "max_conf":    result["max_conf"],
            "mean_conf":   result.get("mean_conf", 0),
            "filename":    file.filename or "camera_capture",
            "mode":        model_mode,
        })

        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/api/status")
def api_status():
    return jsonify({
        "status":    "online",
        "device":    str(device),
        "threshold": threshold,
        "model":     model_label,
        "mode":      model_mode,
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
