"""
SafeWeb — Multi-Algorithm Training Script
=========================================
Run ONCE before starting app.py to enable the Model Lab feature.

    python train_all_models.py

Trains 6 classification algorithms on synthetic URL feature data,
evaluates each one, and saves:
  - model_knn.pkl           K-Nearest Neighbor
  - model_dt.pkl            Decision Tree
  - model_lr.pkl            Logistic Regression
  - model_nb.pkl            Naive Bayes
  - model_rf.pkl            Random Forest
  - model_xgb.pkl           XGBoost  (or model_gb.pkl if XGBoost not installed)
  - all_models_scaler.pkl   Scaler trained on this dataset
  - model_metrics.json      Accuracy, precision, recall, F1 + confusion matrices
"""

import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

# ── Classifiers ────────────────────────────────────────────────────────────────
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not found — using GradientBoostingClassifier instead.")
    print("       To install XGBoost: pip install xgboost\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate synthetic URL feature dataset
# ═══════════════════════════════════════════════════════════════════════════════
# Features (same 9 as the production model):
#   0: url_length      1: domain_length   2: num_dots
#   3: num_hyphens     4: num_at_symbols  5: num_slashes
#   6: num_digits      7: has_https       8: has_ip
# Labels: 0 = Legitimate (good), 1 = Phishing (bad) — matches SafeWeb-code.ipynb

np.random.seed(42)
N = 1500    # samples per class


def make_phishing(n: int) -> np.ndarray:
    """Synthetic phishing URLs — long, no HTTPS, lots of dots/digits, IP use."""
    return np.column_stack([
        np.random.randint(80,  200, n),                          # url_length
        np.random.randint(20,  60,  n),                          # domain_length
        np.random.randint(4,   10,  n),                          # num_dots
        np.random.randint(2,   8,   n),                          # num_hyphens
        np.random.choice([0, 1], n, p=[0.65, 0.35]),             # num_at (35% have @)
        np.random.randint(5,   12,  n),                          # num_slashes
        np.random.randint(8,   22,  n),                          # num_digits
        np.random.choice([0, 1], n, p=[0.80, 0.20]),             # has_https (mostly 0)
        np.random.choice([0, 1], n, p=[0.68, 0.32]),             # has_ip (32% use IP)
    ]).astype(float)


def make_legitimate(n: int) -> np.ndarray:
    """Synthetic legitimate URLs — short, HTTPS, clean structure."""
    return np.column_stack([
        np.random.randint(15,  80,  n),                          # url_length
        np.random.randint(5,   25,  n),                          # domain_length
        np.random.randint(1,   4,   n),                          # num_dots
        np.random.randint(0,   2,   n),                          # num_hyphens
        np.zeros(n, dtype=int),                                  # num_at (never)
        np.random.randint(1,   5,   n),                          # num_slashes
        np.random.randint(0,   5,   n),                          # num_digits
        np.ones(n, dtype=int),                                   # has_https (always)
        np.zeros(n, dtype=int),                                  # has_ip (never)
    ]).astype(float)


X = np.vstack([make_phishing(N), make_legitimate(N)])
y = np.array([1] * N + [0] * N)     # 1=Phishing, 0=Legitimate (same as notebook)

# Train / test split (80% / 20%), stratified to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (required by KNN, LR, NB; helps others too)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "all_models_scaler.pkl")
print("[SAVED] all_models_scaler.pkl\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Define the 6 classifiers
# ═══════════════════════════════════════════════════════════════════════════════

classifiers = [
    # (display_name, file_slug, model_object)
    (
        "K-Nearest Neighbor",
        "knn",
        KNeighborsClassifier(n_neighbors=5),
    ),
    (
        "Decision Tree",
        "dt",
        DecisionTreeClassifier(max_depth=10, random_state=42),
    ),
    (
        "Logistic Regression",
        "lr",
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    ),
    (
        "Naive Bayes",
        "nb",
        GaussianNB(),
    ),
    (
        "Random Forest",
        "rf",
        RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    ),
    (
        "XGBoost" if HAS_XGB else "Gradient Boosting",
        "xgb" if HAS_XGB else "gb",
        XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                      random_state=42, eval_metric="logloss",
                      verbosity=0) if HAS_XGB
        else GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.1, random_state=42),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train, evaluate, and save each model
# ═══════════════════════════════════════════════════════════════════════════════

metrics_output       = {}
feature_importances  = {}
FEATURE_NAMES        = [
    "URL Length", "Domain Length", "Num Dots", "Num Hyphens",
    "Num @ Symbols", "Num Slashes", "Num Digits", "Has HTTPS", "Has IP"
]

print("=" * 55)
print("  SafeWeb — Training 6 Classification Algorithms")
print("=" * 55)

for name, slug, clf in classifiers:
    print(f"\n▶  Training: {name}")

    # ── Train ──────────────────────────────────────────────────────────────
    clf.fit(X_train_s, y_train)

    # ── Evaluate on test set ───────────────────────────────────────────────
    y_pred = clf.predict(X_test_s)

    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    # ── 5-fold cross-validation accuracy ──────────────────────────────────
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="accuracy")
    cv_mean   = float(cv_scores.mean())
    cv_std    = float(cv_scores.std())

    metrics_output[name] = {
        "accuracy":    round(acc  * 100, 2),
        "precision":   round(prec * 100, 2),
        "recall":      round(rec  * 100, 2),
        "f1_score":    round(f1   * 100, 2),
        "cv_accuracy": round(cv_mean * 100, 2),
        "cv_std":      round(cv_std  * 100, 2),
        "confusion_matrix": cm,
    }

    # ── Feature importances (tree-based models only) ───────────────────────
    if hasattr(clf, "feature_importances_"):
        feature_importances[name] = [
            round(v, 4) for v in clf.feature_importances_.tolist()
        ]

    # ── Save model ─────────────────────────────────────────────────────────
    out_path = f"model_{slug}.pkl"
    joblib.dump(clf, out_path)

    print(f"   Accuracy : {acc*100:.1f}%   Precision: {prec*100:.1f}%")
    print(f"   Recall   : {rec*100:.1f}%   F1-Score : {f1*100:.1f}%")
    print(f"   CV Acc   : {cv_mean*100:.1f}% ± {cv_std*100:.1f}%")
    print(f"   Saved →  {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Save all metrics to JSON (used by the dashboard /viz-data endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

# Static dataset-level feature averages (for visualization charts)
avg_phishing  = make_phishing(2000).mean(axis=0).tolist()
avg_legit     = make_legitimate(2000).mean(axis=0).tolist()

output = {
    "metrics":              metrics_output,
    "feature_importances":  feature_importances,
    "feature_names":        FEATURE_NAMES,
    "dataset_averages": {
        "phishing":    [round(v, 2) for v in avg_phishing],
        "legitimate":  [round(v, 2) for v in avg_legit],
    },
    "training_info": {
        "samples_per_class": N,
        "total_samples":     N * 2,
        "test_split":        "20%",
        "cv_folds":          5,
    },
}

with open("model_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 55)
print("[SAVED] model_metrics.json — dashboard visualization data")
print("\n✅  All done! Start the server with:  python app.py")
print("=" * 55 + "\n")