# =============================================================================
# SafeWeb — Enhanced Phishing URL Detection API
# =============================================================================
# HOW TO RUN:
#   1. Install dependencies:
#        pip install flask flask-cors joblib scikit-learn numpy
#        pip install python-whois      (optional — for WHOIS domain age)
#   2. Place in the same directory:
#        phishing_production_model.pkl
#        url_production_scaler.pkl
#   3. (Optional) Run train_all_models.py first for Model Lab feature
#   4. Start the server:
#        python app.py
# =============================================================================

import os
import re
import sys
import json
import time
import socket
import warnings
import datetime
import joblib
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── Optional WHOIS support ──────────────────────────────────────────────────
try:
    import whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Session statistics (in-memory, resets on restart) ───────────────────────
stats = {
    "total":    0,
    "phishing": 0,
    "safe":     0,
    "history":  [],   # last 100 scans with timestamps
}

# ─── Feature metadata ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "URL Length",
    "Domain Length",
    "Num Dots",
    "Num Hyphens",
    "Num @ Symbols",
    "Num Slashes",
    "Num Digits",
    "Has HTTPS",
    "Has IP Address",
]

# Threshold rules: (threshold, operator, label)
#   operator: ">" | "<" | "==" | "!="
FEATURE_THRESHOLDS = [
    (75,  ">",  "Long URL"),
    (30,  ">",  "Long Domain"),
    (3,   ">",  "Many Dots"),
    (2,   ">",  "Many Hyphens"),
    (0,   ">",  "Contains @"),
    (6,   ">",  "Deep Path"),
    (8,   ">",  "Many Digits"),
    (1,   "!=", "No HTTPS"),
    (1,   "==", "IP in URL"),
]

# ─── Load production model ───────────────────────────────────────────────────
MODEL_PATH  = os.path.join(BASE_DIR, "phishing_production_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "url_production_scaler.pkl")

for _path, _name in [(MODEL_PATH,  "phishing_production_model.pkl"),
                     (SCALER_PATH, "url_production_scaler.pkl")]:
    if not os.path.exists(_path):
        print(f"\n[ERROR] Required file not found: {_name}")
        print(f"        Expected at: {_path}\n")
        sys.exit(1)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[OK] Production model and scaler loaded.")

# ─── Load all trained models (optional — created by train_all_models.py) ─────
ALL_MODELS_SCALER_PATH = os.path.join(BASE_DIR, "all_models_scaler.pkl")
all_models_scaler = None
if os.path.exists(ALL_MODELS_SCALER_PATH):
    all_models_scaler = joblib.load(ALL_MODELS_SCALER_PATH)
    print("[OK] Multi-model scaler loaded.")

# Model filenames created by train_all_models.py
TRAINED_MODEL_FILES = {
    "K-Nearest Neighbor":  "model_knn.pkl",
    "Decision Tree":       "model_dt.pkl",
    "Logistic Regression": "model_lr.pkl",
    "Naive Bayes":         "model_nb.pkl",
    "Random Forest":       "model_rf.pkl",
    "XGBoost":             "model_xgb.pkl",     # may not exist
    "Gradient Boosting":   "model_gb.pkl",      # fallback
}

loaded_models = {}
for name, fname in TRAINED_MODEL_FILES.items():
    fpath = os.path.join(BASE_DIR, fname)
    if os.path.exists(fpath):
        loaded_models[name] = joblib.load(fpath)
        print(f"[OK] {name} model loaded.")

# Label convention — must match SafeWeb-code.ipynb and train_all_models.py:
#   0 = Legitimate / Safe (dataset label "good")
#   1 = Phishing / Malicious (dataset label "bad")
LABEL_LEGITIMATE = 0
LABEL_PHISHING = 1


def decode_prediction(clf, scaled_row) -> tuple:
    """Return (human_label, confidence_percent) from a fitted classifier."""
    pred = int(clf.predict(scaled_row)[0])
    probas = clf.predict_proba(scaled_row)[0]
    classes = list(clf.classes_)
    conf = float(probas[classes.index(pred)]) * 100
    label = "Phishing" if pred == LABEL_PHISHING else "Legitimate"
    return label, conf


# ─── Feature extraction ───────────────────────────────────────────────────────
_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)
_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z\d+\-.]*://")


def _normalize_url(url: str) -> str:
    """
    Match training data format: CSV URLs are usually scheme-less.
    Strip http(s):// before parsing so user-entered https does not skew features.
    """
    url = url.strip()
    if not url:
        return "http://"
    url = _SCHEME_PATTERN.sub("", url)
    return "http://" + url.lstrip("/")


def extract_features(url: str) -> list:
    """
    Extract 9 features from a URL:
      1. url_length       — total character count
      2. domain_length    — netloc character count
      3. num_dots         — count of '.' in URL
      4. num_hyphens      — count of '-' in URL
      5. num_at_symbols   — count of '@' in URL
      6. num_slashes      — count of '/' in URL
      7. num_digits       — count of digit characters
      8. has_https        — 1 if HTTPS, else 0
      9. has_ip           — 1 if hostname is an IPv4 address
    """
    original = url.strip()
    url = _normalize_url(url)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    has_https = 1 if _SCHEME_PATTERN.match(original) and original.lower().startswith("https") else 0
    return [
        len(url),
        len(parsed.netloc),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("/"),
        sum(c.isdigit() for c in url),
        has_https,
        1 if _IP_PATTERN.match(hostname) else 0,
    ]


def compute_risk_score(features: list) -> dict:
    """
    Rule-based risk score (0–100) with per-rule breakdown.
    Separate from the ML model — gives explainability.
    """
    rules = []
    score = 0

    val = features[0]   # url_length
    if val > 100:
        pts = 15; score += pts; rules.append({"rule": f"URL length {val} chars (> 100)", "points": pts, "severity": "high"})
    elif val > 75:
        pts = 8; score += pts; rules.append({"rule": f"URL length {val} chars (> 75)", "points": pts, "severity": "medium"})

    val = features[7]   # has_https
    if val == 0:
        pts = 25; score += pts; rules.append({"rule": "No HTTPS encryption detected", "points": pts, "severity": "critical"})

    val = features[4]   # num_at
    if val > 0:
        pts = 20; score += pts; rules.append({"rule": f"Contains @ symbol (credential trick)", "points": pts, "severity": "critical"})

    val = features[8]   # has_ip
    if val == 1:
        pts = 20; score += pts; rules.append({"rule": "IP address used instead of domain", "points": pts, "severity": "critical"})

    val = features[2]   # num_dots
    if val > 5:
        pts = 10; score += pts; rules.append({"rule": f"Excessive dots in URL ({val})", "points": pts, "severity": "high"})
    elif val > 3:
        pts = 5; score += pts; rules.append({"rule": f"Multiple sub-domains ({val} dots)", "points": pts, "severity": "medium"})

    val = features[3]   # num_hyphens
    if val > 3:
        pts = 10; score += pts; rules.append({"rule": f"Many hyphens ({val}) — common in spoof domains", "points": pts, "severity": "high"})

    val = features[6]   # num_digits
    if val > 10:
        pts = 5; score += pts; rules.append({"rule": f"High digit count ({val}) — looks auto-generated", "points": pts, "severity": "low"})

    val = features[1]   # domain_length
    if val > 40:
        pts = 10; score += pts; rules.append({"rule": f"Domain length {val} chars (very long)", "points": pts, "severity": "high"})

    score = min(score, 100)

    if score >= 70:
        level = "CRITICAL"
    elif score >= 45:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"score": score, "level": level, "rules": rules}


def record_stat(url: str, prediction: str, confidence: float, risk_level: str = "—"):
    """Update in-memory session statistics."""
    stats["total"] += 1
    if prediction == "Phishing":
        stats["phishing"] += 1
    else:
        stats["safe"] += 1
    stats["history"].append({
        "url":        url,
        "prediction": prediction,
        "confidence": confidence,
        "risk":       risk_level,
        "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
    })
    if len(stats["history"]) > 100:
        stats["history"].pop(0)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── 1. Single URL prediction ──────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST {"url": "https://example.com"}
    Returns prediction, confidence, raw features, and risk score.
    """
    data = request.get_json(force=True, silent=True)
    if not data or not str(data.get("url", "")).strip():
        return jsonify({"error": "No URL provided"}), 400

    url = str(data["url"]).strip()

    try:
        features = extract_features(url)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 422

    arr    = np.array(features).reshape(1, -1)
    scaled = scaler.transform(arr)
    label, conf = decode_prediction(model, scaled)

    risk = compute_risk_score(features)
    record_stat(url, label, round(conf, 2), risk["level"])

    return jsonify({
        "url":           url,
        "prediction":    label,
        "confidence":    round(conf, 2),
        "features":      features,
        "feature_names": FEATURE_NAMES,
        "risk":          risk,
    })


# ── 2. Batch scan ─────────────────────────────────────────────────────────────
@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    POST {"urls": ["https://a.com", "http://b.com", ...]}
    Returns a list of predictions (same format as /predict).
    """
    data = request.get_json(force=True, silent=True)
    urls = data.get("urls", []) if data else []

    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Provide a list under 'urls'"}), 400

    results = []
    for url in urls[:50]:           # hard cap at 50
        url = str(url).strip()
        if not url:
            continue
        try:
            features = extract_features(url)
            arr    = np.array(features).reshape(1, -1)
            scaled = scaler.transform(arr)
            label, conf = decode_prediction(model, scaled)

            risk = compute_risk_score(features)
            record_stat(url, label, round(conf, 2), risk["level"])

            results.append({
                "url":        url,
                "prediction": label,
                "confidence": round(conf, 2),
                "risk_score": risk["score"],
                "risk_level": risk["level"],
            })
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    phishing_count = sum(1 for r in results if r.get("prediction") == "Phishing")
    return jsonify({
        "results":        results,
        "total":          len(results),
        "phishing_found": phishing_count,
        "safe_found":     len(results) - phishing_count,
    })


# ── 3. Compare all trained models ─────────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare_models():
    """
    POST {"url": "https://example.com"}
    Returns prediction from every loaded model side-by-side.
    """
    data = request.get_json(force=True, silent=True)
    if not data or not str(data.get("url", "")).strip():
        return jsonify({"error": "No URL provided"}), 400

    url      = str(data["url"]).strip()
    features = extract_features(url)
    arr      = np.array(features).reshape(1, -1)

    comparison = {}

    # Production model (uses its own scaler)
    scaled = scaler.transform(arr)
    prod_lbl, prod_conf = decode_prediction(model, scaled)
    comparison["Production Model"] = {
        "prediction": prod_lbl,
        "confidence": round(prod_conf, 2),
    }

    # All other models (use all_models_scaler if available)
    if loaded_models:
        if all_models_scaler:
            arr_s = all_models_scaler.transform(arr)
        else:
            arr_s = scaled    # fallback to production scaler

        for name, clf in loaded_models.items():
            try:
                lbl, conf = decode_prediction(clf, arr_s)
                comparison[name] = {"prediction": lbl, "confidence": round(conf, 2)}
            except Exception as e:
                comparison[name] = {"error": str(e)}

    # Majority vote
    votes_phishing = sum(1 for v in comparison.values() if v.get("prediction") == "Phishing")
    votes_total    = len([v for v in comparison.values() if "prediction" in v])
    majority       = "Phishing" if votes_phishing > votes_total / 2 else "Legitimate"

    return jsonify({
        "url":         url,
        "features":    features,
        "results":     comparison,
        "majority":    majority,
        "votes":       {"phishing": votes_phishing, "legitimate": votes_total - votes_phishing},
        "models_ready": bool(loaded_models),
    })


# ── 4. Session statistics ─────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def get_stats():
    """Returns session scan counts."""
    pct = round(stats["phishing"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
    return jsonify({
        "total":          stats["total"],
        "phishing":       stats["phishing"],
        "safe":           stats["safe"],
        "phishing_pct":   pct,
        "history":        list(reversed(stats["history"])),
        "recent_history": stats["history"][-20:],
    })


# ── 5. Visualization data (from train_all_models.py output) ──────────────────
@app.route("/viz-data", methods=["GET"])
def viz_data():
    """Serves pre-computed model metrics from train_all_models.py."""
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"available": False, "message": "Run train_all_models.py first."})
    with open(metrics_path) as f:
        data = json.load(f)
    data["available"] = True
    if "production_model" not in data:
        data["production_model"] = "Gradient Boosting"
    return jsonify(data)


# ── 6. WHOIS domain age lookup (optional) ────────────────────────────────────
@app.route("/whois", methods=["POST"])
def whois_lookup():
    """
    POST {"url": "https://example.com"}
    Returns domain registration age and registrar info.
    Requires: pip install python-whois
    """
    if not HAS_WHOIS:
        return jsonify({
            "available": False,
            "message":   "Install python-whois: pip install python-whois",
        })

    data = request.get_json(force=True, silent=True)
    if not data or not str(data.get("url", "")).strip():
        return jsonify({"error": "No URL provided"}), 400

    url    = str(data["url"]).strip()
    domain = urlparse(url).netloc or url

    try:
        w          = whois.whois(domain)
        created    = w.creation_date
        if isinstance(created, list):
            created = created[0]

        age_days   = (datetime.datetime.now() - created).days if created else None
        suspicious = age_days is not None and age_days < 30

        return jsonify({
            "available":   True,
            "domain":      domain,
            "created":     str(created) if created else "Unknown",
            "age_days":    age_days,
            "registrar":   w.registrar or "Unknown",
            "country":     w.country or "Unknown",
            "suspicious":  suspicious,
            "note":        "Domain < 30 days old — high phishing risk!" if suspicious else "",
        })
    except Exception as e:
        return jsonify({"available": True, "error": str(e), "domain": domain})


# ─── Entry point ─────────────────────────────────────────────────────────────
def find_free_port(start: int = 5001, attempts: int = 10) -> int:
    """Return the first available port starting from `start`."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


if __name__ == "__main__":
    # Port 5000 is often taken by macOS AirPlay Receiver — default to 5001
    requested = int(os.environ.get("PORT", 5001))

    # Flask debug reloader runs this script twice. Pick the port once in the
    # supervisor process so the child does not see its own port as "busy".
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        if "SAFEWEB_PORT" not in os.environ:
            port = find_free_port(requested) if os.environ.get("PORT") is None else requested
            os.environ["SAFEWEB_PORT"] = str(port)
            if port != requested:
                print(f"[INFO]  Port {requested} is busy — using {port} instead.")

    port = int(os.environ.get("SAFEWEB_PORT", os.environ.get("PORT", requested)))

    # Avoid duplicate startup banners from supervisor + reloader child
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(f"\n[SafeWeb] Server starting on http://localhost:{port}")
        if not loaded_models:
            print("[INFO]  Model Lab disabled — run train_all_models.py to enable it.")
        if not HAS_WHOIS:
            print("[INFO]  WHOIS lookup disabled — run: pip install python-whois")
        print()

    app.run(host="0.0.0.0", port=port, debug=True)