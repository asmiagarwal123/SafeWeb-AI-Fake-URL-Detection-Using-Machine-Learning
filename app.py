# =============================================================================
# SafeWeb — Phishing URL Detection API
# =============================================================================
# HOW TO RUN:
#   1. Install dependencies:
#        pip install flask flask-cors joblib scikit-learn numpy
#   2. Place these files in the same directory as app.py:
#        - phishing_production_model.pkl
#        - url_production_scaler.pkl
#   3. Start the server:
#        python app.py
#   4. The API will be available at http://localhost:5000
#      POST /predict  →  { "url": "https://example.com" }
# =============================================================================

import os
import re
import sys
import warnings
import joblib
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Directory where app.py lives — used to locate index.html and .pkl files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Suppress sklearn version-mismatch warnings (model trained on 1.7.2, running on 1.8.x)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Root route — serves index.html so visiting http://localhost:5000 opens the app
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ---------------------------------------------------------------------------
# Load model artefacts once at startup
# ---------------------------------------------------------------------------
MODEL_PATH  = os.path.join(BASE_DIR, "phishing_production_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "url_production_scaler.pkl")

# ---------------------------------------------------------------------------
# Startup checks — fail fast with a clear message if files are missing
# ---------------------------------------------------------------------------
for _path, _name in [(MODEL_PATH, "phishing_production_model.pkl"),
                     (SCALER_PATH, "url_production_scaler.pkl")]:
    if not os.path.exists(_path):
        print(f"\n[ERROR] Required file not found: {_name}")
        print(f"        Expected at: {_path}")
        print("        Place both .pkl files in the same folder as app.py and try again.\n")
        sys.exit(1)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[OK] Model and scaler loaded successfully.")

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
# Regex to detect an IPv4 address in the hostname
_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)


def extract_features(url: str) -> list:
    """
    Extract exactly 9 features from *url* in this order:
      1. url_length      – total character count of the URL
      2. domain_length   – character count of the netloc (host + port)
      3. num_dots        – count of '.' in the full URL
      4. num_hyphens     – count of '-' in the full URL
      5. num_at_symbols  – count of '@' in the full URL
      6. num_slashes     – count of '/' in the full URL
      7. num_digits      – count of digit characters in the full URL
      8. has_https       – 1 if scheme is 'https', else 0
      9. has_ip          – 1 if the hostname looks like an IPv4 address, else 0
    """
    parsed = urlparse(url)

    url_length    = len(url)
    domain_length = len(parsed.netloc)
    num_dots      = url.count(".")
    num_hyphens   = url.count("-")
    num_at_symbols= url.count("@")
    num_slashes   = url.count("/")
    num_digits    = sum(c.isdigit() for c in url)
    has_https     = 1 if parsed.scheme.lower() == "https" else 0
    # Strip port before IP check
    hostname      = parsed.hostname or ""
    has_ip        = 1 if _IP_PATTERN.match(hostname) else 0

    return [
        url_length,
        domain_length,
        num_dots,
        num_hyphens,
        num_at_symbols,
        num_slashes,
        num_digits,
        has_https,
        has_ip,
    ]


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: {"url": "https://example.com"}
    Returns: {"url": "...", "prediction": "Phishing" | "Legitimate", "confidence": 94.2}
    """
    data = request.get_json(force=True, silent=True)

    # --- Input validation ---
    if not data or "url" not in data or not str(data["url"]).strip():
        return jsonify({"error": "No URL provided"}), 400

    url = str(data["url"]).strip()

    # --- Feature extraction ---
    try:
        features = extract_features(url)
    except Exception as exc:
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 422

    # --- Scale & predict ---
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction_label = model.predict(features_scaled)[0]
    probabilities    = model.predict_proba(features_scaled)[0]

    # -----------------------------------------------------------------------
    # Label convention (confirmed from training data):
    #   0 → Phishing    (proba index 0)
    #   1 → Legitimate  (proba index 1)
    # -----------------------------------------------------------------------
    if int(prediction_label) == 0:
        label      = "Phishing"
        confidence = float(probabilities[0]) * 100   # confidence in being phishing
    else:
        label      = "Legitimate"
        confidence = float(probabilities[1]) * 100   # confidence in being legitimate

    return jsonify(
        {
            "url":        url,
            "prediction": label,
            "confidence": round(confidence, 2),
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
