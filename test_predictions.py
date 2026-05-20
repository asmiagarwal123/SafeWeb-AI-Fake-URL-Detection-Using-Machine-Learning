import os
import joblib
import numpy as np
from urllib.parse import urlparse
import re

BASE_DIR = r"c:\Users\asmi\Desktop\Projects\URL Phishing\SafeWeb-AI-Fake-URL-Detection-Using-Machine-Learning"

url = "http://paypal1-login-security.com"

# Feature extraction
_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

def extract_features(url: str) -> list:
    url_lower = url.lower()
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        url = "http://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    domain = parsed.netloc
    return [
        len(url),
        len(domain),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("/"),
        sum(c.isdigit() for c in url),
        1 if parsed.scheme.lower() == "https" else 0,
        1 if _IP_PATTERN.match(hostname or domain) else 0,
    ]

features = extract_features(url)
print("URL:", url)
print("Features:", features)

# Test production model
model_path = os.path.join(BASE_DIR, "phishing_production_model.pkl")
scaler_path = os.path.join(BASE_DIR, "url_production_scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    arr = np.array(features).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)[0]
    probas = model.predict_proba(scaled)[0]
    print(f"Production Model -> pred: {pred}, probas: {probas}")

# Test other models
all_models_scaler_path = os.path.join(BASE_DIR, "all_models_scaler.pkl")
if os.path.exists(all_models_scaler_path):
    all_models_scaler = joblib.load(all_models_scaler_path)
    arr_s = all_models_scaler.transform(arr)
    
    TRAINED_MODEL_FILES = {
        "K-Nearest Neighbor":  "model_knn.pkl",
        "Decision Tree":       "model_dt.pkl",
        "Logistic Regression": "model_lr.pkl",
        "Naive Bayes":         "model_nb.pkl",
        "Random Forest":       "model_rf.pkl",
        "XGBoost":             "model_xgb.pkl",
        "Gradient Boosting":   "model_gb.pkl",
    }
    
    for name, fname in TRAINED_MODEL_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            clf = joblib.load(fpath)
            p = clf.predict(arr_s)[0]
            proba = clf.predict_proba(arr_s)[0]
            print(f"{name} -> pred: {p}, probas: {proba}")
