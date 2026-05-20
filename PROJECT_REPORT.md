# SAFEWEB — FAKE URL DETECTION USING MACHINE LEARNING

**Bachelor of Technology**  
in  
**Artificial Intelligence and Machine Learning**

Submitted by  
**Ananya Srivastava (PRN: 24070722003)**

For the Fulfillment of  
**Supervised Machine Learning**

**Symbiosis Institute of Technology, Hyderabad**  
Survey Number: 292, Off Bangalore Highway, Modallaguda Village,  
Nandigama Mandal, Ranga Reddy DIST, Near Hyderabad, Telangana — 509217.

---

## CONTENTS

| Section | Page |
|---------|------|
| Abstract | 2 |
| Introduction | 3 |
| Problem Statement | 4 |
| Literature Survey | 4–6 |
| Dataset Description | 6–8 |
| Data Preprocessing | 8–10 |
| Exploratory Data Analysis | 10–12 |
| Methodology | 12–14 |
| Model Implementation | 14–19 |
| Hyperparameter Tuning | 19–20 |
| Performance Evaluation | 20–22 |
| Results and Comparisons | 22–24 |
| Deployment / Web GUI | 24–26 |
| Conclusion | 26 |
| Future Scope | 27 |
| References | 27 |

---

## ABSTRACT

**SafeWeb** is a machine-learning-powered phishing and fake URL detection system that classifies web addresses as **Legitimate** or **Phishing/Malicious** using lexical URL features extracted without visiting the target page. The project combines a Jupyter Notebook research pipeline (`SafeWeb-code.ipynb`) with a production **Flask REST API** and a rich **web dashboard** (`index.html`) for real-time scanning, batch analysis, model comparison, and risk scoring.

Unlike content-based or browser-based detectors, this system analyzes structural properties of URLs—length, domain structure, special characters, HTTPS usage, and IP-based hostnames—to capture patterns commonly used in credential-harvesting and brand-impersonation attacks. A dataset of **549,346 labeled URLs** is used for training; a stratified random sample of **10,000 records** drives model development in the notebook. Nine numerical features are engineered per URL, standardized with `StandardScaler`, and fed into six supervised classifiers: Logistic Regression, Naïve Bayes, K-Nearest Neighbors, Decision Tree, Support Vector Machine (RBF kernel), and Random Forest.

The primary functionalities of the system are:

- Loading, exploring, and visualizing a large-scale phishing URL dataset.
- Lexical feature extraction and preprocessing (scaling, train/test split).
- Training and comparing multiple classification algorithms with cross-validation.
- Evaluating models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
- Feature importance analysis via Random Forest and Decision Tree visualization.
- Serializing the best model for deployment (`phishing_production_model.pkl`, `url_production_scaler.pkl`).
- Serving predictions through a Flask API with batch scan, model lab, WHOIS lookup, and rule-based risk scoring.
- Interactive dashboard with live statistics, visualizations, and CSV export.

The project demonstrates the full machine learning lifecycle—from dataset acquisition and EDA to model selection, evaluation, serialization, and deployment—using Python libraries including **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, **Seaborn**, **Joblib**, **Flask**, and **Chart.js**. It provides a practical foundation for building cybersecurity decision-support tools that protect users before they click malicious links.

---

## INTRODUCTION

### Background and Motivation

Phishing remains one of the most prevalent and effective cyberattack vectors worldwide. Attackers impersonate trusted brands—banks, payment gateways, social networks, and government portals—using URLs that appear legitimate at a glance. Victims who click these links may expose credentials, financial data, or install malware. Traditional defenses such as blocklists and email filters struggle to keep pace with the volume and velocity of newly registered phishing domains.

Because URLs encode rich structural information (length, subdomains, obfuscation characters, protocol choice), **lexical URL analysis** offers a fast, lightweight alternative that does not require downloading page content. Machine learning can learn subtle combinations of these features from historical labeled data and generalize to unseen URLs. With growing awareness of AI-driven security tools, building an end-to-end phishing detector is both academically relevant and practically valuable.

This project, **SafeWeb**, implements such a system: raw URLs are transformed into numerical feature vectors, multiple classifiers are trained and compared, the best performer is saved, and predictions are exposed through a modern web interface suitable for demonstration, coursework submission, and further extension.

### Objectives

The objectives of this project are:

1. **Acquire and understand** a real-world phishing URL dataset with binary labels (legitimate vs. malicious).
2. **Engineer lexical features** from URL strings using Python's `urllib.parse` and regular expressions.
3. **Preprocess** data through stratified splitting and feature standardization to prepare inputs for distance- and margin-based models.
4. **Train and compare** six supervised classification algorithms using consistent evaluation metrics.
5. **Select the best model** by F1-Score (balancing precision and recall under class imbalance) and serialize it for inference.
6. **Deploy** the trained model via a Flask backend and interactive HTML/JavaScript dashboard for real-time URL scanning.
7. **Extend** the system with rule-based risk scoring, multi-model comparison (Model Lab), batch scanning, and optional WHOIS domain-age lookup.

Working on SafeWeb provided hands-on experience with feature engineering for unstructured text-like inputs, handling imbalanced classification, cross-validation, ROC analysis, model interpretability, and ML deployment—core competencies in applied machine learning and cybersecurity.

---

## PROBLEM STATEMENT

The problem is to **classify a given URL as Phishing (malicious/fake) or Legitimate (safe)** based solely on lexical and structural URL properties, without fetching the webpage content.

**Formal definition:**

- **Input:** A URL string (e.g., `https://secure-login.paypal-verify.example.com/signin`)
- **Output:** A binary class label and confidence score
  - Class **0 — Safe / Legitimate** (mapped from dataset label `good`)
  - Class **1 — Phishing / Malicious** (mapped from dataset label `bad`)
- **Task type:** Supervised binary classification

**Constraints and requirements:**

- Features must be extractable in milliseconds (suitable for real-time scanning).
- The model must generalize to URLs not seen during training.
- False negatives (missing phishing URLs) are especially costly; recall on the phishing class is a priority metric alongside F1-Score.
- The deployed system must expose predictions through an accessible web interface for non-technical users.

---

## LITERATURE SURVEY

Research on phishing URL detection spans lexical analysis, host-based features, content analysis, and hybrid approaches. The following works represent relevant trends and inform the design of SafeWeb.

### Key Research Directions

**1. Lexical and Host-Based Feature Engineering**

Early and widely cited work by Ma et al. and subsequent studies demonstrate that URL length, number of dots, presence of IP addresses, `@` symbols, and HTTPS usage are strong discriminators between phishing and legitimate URLs. Lexical-only approaches remain popular because they are fast, privacy-preserving, and do not depend on page availability.

**2. Ensemble and Boosting Classifiers**

Random Forest, Gradient Boosting, and XGBoost consistently rank among top performers on tabular URL feature datasets. Ensemble methods capture non-linear interactions (e.g., long URL + many dots + no HTTPS) that linear models miss. SafeWeb's notebook and `train_all_models.py` both employ Random Forest and tree ensembles for this reason.

**3. Class Imbalance Handling**

Phishing datasets often contain more legitimate than malicious URLs (or vice versa depending on source). Techniques include SMOTE oversampling, class weighting, stratified splitting, and optimizing F1-Score rather than accuracy alone. SafeWeb uses **stratified 80/20 splitting** and prioritizes **F1-Score** and **Recall** for model selection.

**4. Deep Learning and Character-Level Models**

Recent work applies CNNs, RNNs, and Transformers directly to URL character sequences. While these can achieve high accuracy, they require more compute and training data. SafeWeb deliberately uses interpretable lexical features and classical ML for transparency and deployment simplicity—aligned with educational and prototyping goals.

**5. Explainability and Hybrid Scoring**

Explainable AI (feature importance, SHAP, rule-based overlays) improves trust in security tools. SafeWeb combines **ML predictions** with a separate **rule-based risk score** (0–100) that flags HTTPS absence, `@` symbols, IP hostnames, and excessive URL length—giving users human-readable reasons for a verdict.

### Relation to This Project

| Aspect | Literature Trend | SafeWeb Implementation |
|--------|------------------|------------------------|
| Features | Lexical + host-based | 9 lexical features via `extract_features()` |
| Models | RF, XGBoost, SVM, KNN common | 6 classifiers in notebook; 6 in Model Lab script |
| Imbalance | SMOTE, class weights, stratified split | Stratified split; F1-driven selection |
| Evaluation | Precision, Recall, F1, ROC-AUC | Full metric suite + confusion matrices |
| Deployment | Web apps, browser extensions | Flask API + SafeWeb dashboard |
| Explainability | SHAP, feature importance | RF importance plots; rule-based risk breakdown |

SafeWeb aligns with the **end-to-end applied ML pipeline** described in recent surveys: transparent preprocessing, multi-model benchmarking, held-out test evaluation, and deployment—with a focus on reproducibility suitable for a supervised ML course project.

---

## DATASET DESCRIPTION

### Primary Dataset: Phishing Site URLs

| Property | Value |
|----------|-------|
| **File** | `phishing_site_urls.csv` |
| **Total records** | 549,346 |
| **Columns** | 2 (`URL`, `Label`) |
| **Label: `good`** | 392,924 (71.5%) — Legitimate URLs |
| **Label: `bad`** | 156,422 (28.5%) — Phishing / malicious URLs |
| **Format** | Raw URL strings without protocol prefix in many rows |

**Table 1 — Dataset Schema**

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `URL` | String | Web address or partial path | `www.dghjdgf.com/paypal.co.uk/...` |
| `Label` | String | Class label | `good` or `bad` |

**Table 2 — Sample Records from Dataset**

| # | URL (truncated) | Label |
|---|-----------------|-------|
| 1 | `nobell.it/70ffb52d.../login.SkyPe.com/...` | bad |
| 2 | `serviciosbys.com/paypal.cgi.bin.get-into...` | bad |
| 3 | `docs.google.com/spreadsheet/viewform?...` | bad |
| 4 | `google.com` | good |
| 5 | `youtube.com` | good |

> **Note:** The full dataset contains URLs impersonating PayPal, Apple, banking portals, and other brands, as well as legitimate domains. Some entries contain encoding artifacts; the feature extractor handles malformed URLs via try/except fallback to zero features.

### Working Sample for Notebook Training

To keep notebook execution tractable while preserving class proportions, a **random stratified sample of 10,000 URLs** is drawn (`random_state=42`):

| Class | Count in Sample | Percentage |
|-------|-----------------|------------|
| Safe (`good` → 0) | 7,143 | 71.4% |
| Phishing (`bad` → 1) | 2,857 | 28.6% |

**Label encoding used in modeling:**

```
good  →  0  (Safe / Legitimate)
bad   →  1  (Phishing / Malicious)
```

### Supplementary Dataset (Model Lab)

The script `train_all_models.py` generates a **synthetic balanced dataset** (3,000 samples: 1,500 phishing + 1,500 legitimate) using domain-informed random ranges for each feature. This supports the **Model Lab** comparison UI when full retraining on the CSV is impractical at demo time. Synthetic ranges mirror real patterns: phishing URLs tend to be longer, use more dots/digits, lack HTTPS, and sometimes use IP addresses.

**Table 3 — Synthetic vs. Real Feature Averages (Model Lab)**

| Feature | Phishing (avg) | Legitimate (avg) |
|---------|----------------|------------------|
| URL Length | 139.54 | 47.39 |
| Domain Length | 39.98 | 14.35 |
| Num Dots | 6.54 | 2.03 |
| Num Hyphens | 4.55 | 0.49 |
| Num @ Symbols | 0.38 | 0.00 |
| Num Slashes | 8.00 | 2.52 |
| Num Digits | 14.60 | 2.01 |
| Has HTTPS | 0.20 | 1.00 |
| Has IP | 0.32 | 0.00 |

---

## DATA PREPROCESSING

Preprocessing transforms raw URL strings into a clean numeric feature matrix ready for scikit-learn estimators. All steps are implemented in **Phase 3–4** of `SafeWeb-code.ipynb` and mirrored in `app.py` for inference.

### Step 1 — Label Standardization

Categorical labels are mapped to integers:

```python
df['Label'] = df['Label'].map({'good': 0, 'bad': 1})
```

### Step 2 — Sampling

A working subset is created for training efficiency:

```python
df_sample = df.sample(n=10000, random_state=42).copy()
```

### Step 3 — Lexical Feature Extraction

The function `extract_features(url)` parses each URL and returns **9 numerical features**:

| # | Feature | Description | Extraction Logic |
|---|---------|-------------|------------------|
| 1 | `url_length` | Total character count | `len(url)` |
| 2 | `domain_length` | Hostname length | `len(parsed.netloc)` |
| 3 | `num_dots` | Dot count | `url.count('.')` |
| 4 | `num_hyphens` | Hyphen count | `url.count('-')` |
| 5 | `num_at_symbols` | `@` symbol count | `url.count('@')` |
| 6 | `num_slashes` | Path depth indicator | `url.count('/')` |
| 7 | `num_digits` | Numeric character count | `sum(c.isdigit() for c in url)` |
| 8 | `has_https` | Secure protocol flag | `1` if scheme is `https`, else `0` |
| 9 | `has_ip` | IP-based hostname flag | `1` if hostname matches IPv4 regex |

**Preprocessing detail:** URLs missing a scheme are prefixed with `http://` before parsing. Invalid URLs return a zero vector to prevent pipeline crashes.

### Step 4 — Feature Matrix Construction

```python
features_list = df_sample['URL'].apply(extract_features).tolist()
X = pd.DataFrame(features_list, columns=feature_columns)
y = df_sample['Label'].values
```

### Step 5 — Train–Test Split

An **80/20 split** is applied with fixed random seed for reproducibility:

| Set | Samples |
|-----|---------|
| Training | 8,000 |
| Testing | 2,000 |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 6 — Feature Scaling

`StandardScaler` is fit **only on training data** and applied to test data—preventing data leakage:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling is essential for **KNN**, **SVM**, and **Logistic Regression**, which are sensitive to feature magnitude. Tree-based models are scale-invariant but still benefit from consistent pipeline design.

**Table 4 — Mean Feature Values by Class (10K Sample)**

| Feature | Safe (0) | Phishing (1) |
|---------|----------|--------------|
| url_length | 52.79 | 70.23 |
| domain_length | 15.81 | 20.25 |
| num_dots | 1.77 | 2.75 |
| num_hyphens | 1.34 | 0.63 |
| num_at_symbols | 0.00 | 0.01 |
| num_slashes | 4.36 | 4.71 |
| num_digits | 3.10 | 8.63 |
| has_https | 0.00 | 0.00 |
| has_ip | 0.00 | 0.04 |

Phishing URLs in this sample tend to be **longer**, contain **more digits**, and slightly **more dots**—consistent with obfuscation tactics.

> **[INSERT SCREENSHOT HERE]**  
> *Figure 1 — Feature matrix preview (`X.head()`) from Jupyter Notebook after feature extraction.*

---

## EXPLORATORY DATA ANALYSIS

EDA (Phase 2 of the notebook) examines class balance and URL length distributions to guide metric choice and preprocessing.

### 1. Class Balance

The dataset exhibits **moderate class imbalance**: approximately **71% safe** vs. **29% phishing** in the 10K sample. This motivates:

- Using **F1-Score** as the primary comparison metric (not accuracy alone).
- Monitoring **Recall** on the phishing class to minimize missed attacks.
- Considering stratified splitting to preserve proportions in train and test sets.

> **[INSERT SCREENSHOT HERE]**  
> *Figure 2 — Class Balance pie chart (Safe vs. Phishing) from notebook EDA cell.*

### 2. URL Length Distribution

A histogram of URL length by class reveals that phishing URLs often occupy the **upper tail** of the length distribution (many characters, deep paths, tracking parameters), while legitimate URLs cluster at shorter lengths.

> **[INSERT SCREENSHOT HERE]**  
> *Figure 3 — URL Length Distribution histogram (green = Safe, red = Phishing) from notebook EDA.*

### 3. Feature Importance Preview (Post-Training)

After Random Forest training, **`url_length`**, **`domain_length`**, and **`num_digits`** emerge as top contributors—confirming EDA intuitions about structural complexity.

**Table 5 — Random Forest Feature Importance (Notebook Run)**

| Feature | Importance |
|---------|------------|
| url_length | 0.2991 |
| domain_length | 0.2051 |
| num_digits | 0.1747 |
| num_dots | 0.1248 |
| num_slashes | 0.1159 |
| num_hyphens | 0.0651 |
| has_ip | 0.0114 |
| num_at_symbols | 0.0039 |
| has_https | 0.0000 |

> **[INSERT SCREENSHOT HERE]**  
> *Figure 4 — Random Forest Feature Importance horizontal bar chart from notebook Phase 9.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 5 — Decision Tree logic pathway (truncated to depth 3) from notebook Phase 9.*

---

## METHODOLOGY

SafeWeb follows a modular **end-to-end machine learning pipeline** implemented primarily in `SafeWeb-code.ipynb`, with deployment artifacts consumed by `app.py` and `train_all_models.py`.

### System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ phishing_site   │────▶│ Feature          │────▶│ StandardScaler  │
│ _urls.csv       │     │ Extraction (9)   │     │ (fit on train)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │
                    ┌───────────────────────────────────────┘
                    ▼
         ┌──────────────────────┐     ┌──────────────────────┐
         │ 6 Classifiers        │────▶│ Evaluation Metrics   │
         │ (LR, NB, KNN, DT,    │     │ Acc, Prec, Rec, F1,  │
         │  SVM, RF)            │     │ ROC-AUC, CM          │
         └──────────┬───────────┘     └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐     ┌──────────────────────┐
         │ Best Model + Scaler    │────▶│ Flask API + SafeWeb  │
         │ (.pkl files)           │     │ Dashboard (index.html)│
         └──────────────────────┘     └──────────────────────┘
```

### Pipeline Modules

| Module | File | Responsibility |
|--------|------|----------------|
| Data Acquisition | `SafeWeb-code.ipynb` Phase 1 | Load CSV, encode labels, sample |
| EDA | Phase 2 | Distribution plots |
| Feature Engineering | Phase 3 | `extract_features()` |
| Preprocessing | Phase 4 | Split, scale |
| Cross-Validation | Phase 5 | 5-fold CV stability check |
| Model Training | Phase 6 | Fit 6 classifiers |
| ROC Analysis | Phase 7 | AUC curves |
| Evaluation | Phase 8 | Metric comparison table |
| Interpretability | Phase 9 | RF importance, DT plot |
| Serialization | Phase 10 | Save `.pkl` files |
| API Server | `app.py` | REST endpoints |
| Frontend | `index.html` | Dashboard UI |
| Model Lab Training | `train_all_models.py` | Multi-algorithm `.pkl` + metrics JSON |

### Design Approach

1. **Modular notebook cells** — Each phase runs independently for debugging and demonstration.
2. **Held-out test set** — 20% of data is never used during training or CV on the full train split.
3. **Consistent feature order** — The same 9 features in the same order flow from notebook → scaler → API → UI.
4. **Dual scoring** — ML classification plus independent rule-based risk score for explainability.

### Tools and Technologies

| Category | Technology |
|----------|------------|
| Language | Python 3.x |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost (optional) |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Model persistence | Joblib |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, JavaScript |
| Optional enrichment | python-whois |
| IDE | Jupyter Notebook, VS Code / Cursor |

---

## MODEL IMPLEMENTATION

Six supervised classification algorithms are implemented, trained on `X_train_scaled`, and evaluated on `X_test_scaled`. Below is a concise description of each, aligned with the notebook code.

### 1) Logistic Regression

Logistic Regression models the probability of the phishing class using the sigmoid function over a linear combination of features. It serves as a strong, interpretable baseline.

**Implementation:**
```python
LogisticRegression(max_iter=1000, random_state=42)
```

Features are scaled via `StandardScaler`. Coefficients indicate direction of association between URL properties and phishing likelihood.

---

### 2) Naïve Bayes (GaussianNB)

A probabilistic classifier assuming conditional feature independence. Extremely fast to train; useful as a lightweight baseline for URL feature vectors.

**Implementation:**
```python
GaussianNB()
```

---

### 3) K-Nearest Neighbors (KNN)

Classifies a URL by majority vote among the **K=5** nearest training points in scaled feature space. Captures local non-linear boundaries.

**Implementation:**
```python
KNeighborsClassifier(n_neighbors=5)
```

Feature scaling is **mandatory** because KNN relies on Euclidean distance.

> **[INSERT SCREENSHOT HERE]**  
> *Figure 6 — KNN Confusion Matrix from notebook Phase 6 (2×3 grid of all model confusion matrices).*

---

### 4) Decision Tree

A rule-based model that recursively splits on feature thresholds. Limited to `max_depth=10` to reduce overfitting.

**Implementation:**
```python
DecisionTreeClassifier(random_state=42, max_depth=10)
```

Highly interpretable; visualized with `plot_tree` truncated to depth 3.

---

### 5) Support Vector Machine (SVM — RBF Kernel)

Finds a maximum-margin boundary in a kernel-transformed space. Effective for non-linear separation.

**Implementation:**
```python
SVC(kernel='rbf', probability=True, random_state=42)
```

`probability=True` enables ROC-AUC computation via `predict_proba`.

---

### 6) Random Forest

An ensemble of decision trees trained on bootstrap samples with random feature subsets. Robust to noise and captures feature interactions.

**Implementation:**
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

Provides `feature_importances_` for interpretability. **Selected as the best model** by F1-Score in the notebook pipeline run documented in this report.

---

### Model Training Loop (Notebook Phase 6)

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
```

> **[INSERT SCREENSHOT HERE]**  
> *Figure 7 — Full 2×3 Confusion Matrix grid for all six models (notebook Phase 6 output).*

---

## HYPERPARAMETER TUNING

### Cross-Validation (Phase 5)

Before full evaluation, **5-fold cross-validation** on the training set assesses model stability:

| Model | CV Accuracy (mean) | Notes |
|-------|-------------------|-------|
| Logistic Regression | ~77.31% | Stable linear baseline |
| Random Forest | ~82.37% | Lowest variance among strong performers |

```python
cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
```

Low variance across folds indicates models are not overfitting severely to a particular split.

### Model Lab Script Hyperparameters (`train_all_models.py`)

The Model Lab training script uses fixed hyperparameters tuned for the synthetic dataset:

| Algorithm | Key Hyperparameters |
|-----------|---------------------|
| KNN | `n_neighbors=5` |
| Decision Tree | `max_depth=10`, `random_state=42` |
| Logistic Regression | `max_iter=1000`, `C=1.0` |
| Naïve Bayes | Default Gaussian priors |
| Random Forest | `n_estimators=150`, `max_depth=12` |
| XGBoost | `n_estimators=150`, `max_depth=6`, `learning_rate=0.1` |

**Future tuning** could apply `GridSearchCV` on `n_estimators`, `max_depth`, and `n_neighbors` using F1-Score as the scoring metric—similar to best practices in ensemble optimization literature.

---

## PERFORMANCE EVALUATION

All models are evaluated on the **held-out 2,000-sample test set** from the 10K notebook sample. Predictions yield four outcomes:

| | Predicted Safe | Predicted Phishing |
|---|----------------|-------------------|
| **Actual Safe** | True Negative (TN) | False Positive (FP) |
| **Actual Phishing** | False Negative (FN) | True Positive (TP) |

### Metrics Defined

| Metric | Formula | Relevance to Phishing Detection |
|--------|---------|--------------------------------|
| **Accuracy** | (TP+TN) / Total | Overall correctness; misleading under imbalance |
| **Precision** | TP / (TP+FP) | Trustworthiness of phishing alerts |
| **Recall** | TP / (TP+FN) | Ability to catch actual phishing URLs |
| **F1-Score** | 2·(Prec·Rec)/(Prec+Rec) | Primary balance metric for this project |
| **ROC-AUC** | Area under ROC curve | Threshold-independent separability |
| **Confusion Matrix** | 2×2 count matrix | Visual error pattern analysis |

In a security context, **false negatives** (phishing classified as safe) are the most dangerous errors; however, excessive false positives erode user trust. **F1-Score** balances both concerns.

### ROC Curve Analysis (Phase 7)

ROC curves plot True Positive Rate vs. False Positive Rate at varying thresholds. Models with higher AUC discriminate better between classes.

> **[INSERT SCREENSHOT HERE]**  
> *Figure 8 — ROC Curve comparison for all six models with AUC values in legend (notebook Phase 7).*

---

## RESULTS AND COMPARISONS

### Target Distribution Summary

The 10,000-record working sample maintains the original dataset's imbalance (~71% safe, ~29% phishing). This validates the need for F1-Score and recall-aware model selection rather than accuracy alone.

### Model Comparison — Test Set Results

**Table 6 — Comprehensive Model Performance (80/20 Split, 10K Sample)**

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | 5-Fold CV Acc (%) |
|-------|-------------|---------------|------------|--------------|-------------|-------------------|
| **Random Forest** | **83.55** | **76.51** | **61.74** | **68.33** | **89.27** | **82.37** |
| KNN | 82.40 | 74.19 | 59.48 | 66.02 | 85.98 | 81.91 |
| Decision Tree | 81.45 | 82.48 | 45.04 | 58.27 | 79.66 | 80.68 |
| SVM (RBF) | 80.15 | 93.63 | 33.22 | 49.04 | 82.42 | 78.71 |
| Logistic Regression | 78.10 | 83.09 | 29.91 | 43.99 | 73.33 | 77.31 |
| Naïve Bayes | 77.50 | 95.62 | 22.78 | 36.80 | 73.29 | 76.50 |

> **[INSERT SCREENSHOT HERE]**  
> *Figure 9 — Styled comparison table (`df_compare`) with gradient highlighting from notebook Phase 8.*

### Best Performing Model

After evaluating all classifiers, **Random Forest** achieves the highest **F1-Score (68.33%)** and **ROC-AUC (89.27%)** on the test split:

- **Precision:** 76.51% — When the model flags phishing, it is correct roughly three quarters of the time.
- **Recall:** 61.74% — Detects about 62% of phishing URLs in the test set.
- **Cross-validation accuracy:** 82.37% — Consistent generalization across folds.

The model is serialized as:

```
phishing_production_model.pkl
url_production_scaler.pkl
```

### Key Observations

1. **Tree ensembles (Random Forest, KNN)** outperform linear and naive baselines on F1-Score for this feature set.
2. **SVM and Naïve Bayes** achieve very high precision but low recall—they are conservative, missing many phishing URLs.
3. **Decision Tree** balances precision and recall moderately but overfits less than a full-depth tree thanks to `max_depth=10`.
4. **`url_length`**, **`domain_length`**, and **`num_digits`** dominate Random Forest importance—validating lexical complexity as the primary signal.

### Model Lab Results (Synthetic Dataset — `model_metrics.json`)

When trained on the balanced synthetic dataset in `train_all_models.py`, all six algorithms achieve **100% test accuracy** due to well-separated synthetic clusters. These metrics power the **Visualizations** and **Model Lab** dashboard sections.

**Table 7 — Model Lab Algorithm Accuracy (Synthetic 3K Dataset)**

| Algorithm | Accuracy | Precision | Recall | F1 | CV Accuracy |
|-----------|----------|-----------|--------|-----|-------------|
| K-Nearest Neighbor | 100% | 100% | 100% | 100% | 100% |
| Decision Tree | 100% | 100% | 100% | 100% | 100% |
| Logistic Regression | 100% | 100% | 100% | 100% | 100% |
| Naïve Bayes | 100% | 100% | 100% | 100% | 99.96% |
| Random Forest | 100% | 100% | 100% | 100% | 100% |
| XGBoost | 100% | 100% | 100% | 100% | 100% |

> **[INSERT SCREENSHOT HERE]**  
> *Figure 10 — Algorithm Accuracy Comparison bar chart from SafeWeb Dashboard → Visualizations tab.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 11 — Metrics Radar chart (Accuracy, Precision, Recall, F1) from Visualizations tab.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 12 — Confusion Matrix display for Production Model from Visualizations tab.*

---

## DEPLOYMENT / WEB GUI

The deployed system consists of a **Flask REST API** (`app.py`) serving a single-page **SafeWeb Phishing Intelligence Dashboard** (`index.html`).

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves dashboard HTML |
| `/predict` | POST | Single URL ML prediction + risk score |
| `/batch` | POST | Scan up to 50 URLs |
| `/compare` | POST | Compare all loaded models side-by-side |
| `/stats` | GET | Session scan statistics and history |
| `/viz-data` | GET | Model metrics JSON for charts |
| `/whois` | POST | Optional domain age lookup |

### Inference Pipeline (`/predict`)

1. Receive URL JSON payload.
2. Extract 9 features via `extract_features()`.
3. Scale with `url_production_scaler.pkl`.
4. Predict class and probability with `phishing_production_model.pkl`.
5. Compute rule-based risk score (0–100) with per-rule breakdown.
6. Return prediction, confidence, features, and risk object.

### Rule-Based Risk Scoring (Explainability Layer)

Independent of ML, a weighted rule engine assigns points:

| Rule | Points | Severity |
|------|--------|----------|
| No HTTPS | +25 | Critical |
| `@` symbol in URL | +20 | Critical |
| IP address as hostname | +20 | Critical |
| URL length > 100 | +15 | High |
| Domain length > 40 | +10 | High |
| Excessive dots (> 5) | +10 | High |
| Many hyphens (> 3) | +10 | High |
| URL length > 75 | +8 | Medium |
| Multiple subdomains (3–5 dots) | +5 | Medium |
| High digit count (> 10) | +5 | Low |

**Risk levels:** LOW (0–19) · MEDIUM (20–44) · HIGH (45–69) · CRITICAL (70+)

### Dashboard Sections

| Section | Functionality |
|---------|---------------|
| **Dashboard** | Single URL scan, confidence ring, feature breakdown, risk panel, WHOIS |
| **Batch Scan** | Multi-URL paste, summary counts, results table, CSV export |
| **Scan History** | Searchable session log, export, clear |
| **Live Statistics** | Donut chart, confidence histogram, activity feed |
| **Model Lab** | 6 algorithm cards, live multi-model comparison, majority vote |
| **Visualizations** | Feature averages, accuracy bars, feature importance, radar, confusion matrix |
| **Risk Analyzer** | Standalone rule-based risk gauge + ML overlay |

### How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train notebook → saves production .pkl files
# (Run SafeWeb-code.ipynb Phase 10)

# Optional: train Model Lab models
python train_all_models.py

# Start server
python app.py
# Open http://localhost:5001
```

---

### GUI Screenshots (Placeholders)

> **[INSERT SCREENSHOT HERE]**  
> *Figure 13 — SafeWeb Dashboard home: URL Scanner with sidebar navigation (Dashboard, Batch Scan, Scan History, Live Statistics, Model Lab, Visualizations, Risk Analyzer).*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 14 — Phishing detection result: red "PHISHING DETECTED" verdict, confidence ring, feature breakdown with suspicious flags highlighted.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 15 — Legitimate URL result: green "LEGITIMATE URL" verdict with high confidence percentage.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 16 — Feature Breakdown panel showing URL Length, Domain Length, Num Dots, HTTPS status with ✓/⚠ indicators.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 17 — Risk Score Breakdown panel with progress bar and triggered rules (e.g., "No HTTPS +25 pts").*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 18 — WHOIS Domain Information panel (domain age, registrar, suspicious < 30 days warning).*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 19 — Batch Scan: textarea input, summary cards (Total / Phishing / Safe), results table.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 20 — Scan History table with search, Export CSV, and Clear History buttons.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 21 — Live Statistics: four stat cards + donut chart + confidence range bar chart.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 22 — Model Lab: six algorithm cards with accuracy + Live Model Comparison with majority vote banner.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 23 — Visualizations: Phishing vs Legitimate feature averages grouped bar chart.*

> **[INSERT SCREENSHOT HERE]**  
> *Figure 24 — Risk Analyzer: risk level gauge (LOW/MEDIUM/HIGH/CRITICAL) with needle position and rule list.*

---

## CONCLUSION

The **SafeWeb Fake URL Detection** system successfully demonstrates a complete supervised machine learning pipeline applied to cybersecurity. Starting from a corpus of over **549,000 labeled URLs**, the project engineers **nine lexical features**, preprocesses data with stratified splitting and standardization, trains **six classification algorithms**, and rigorously evaluates them using precision, recall, F1-Score, ROC-AUC, and confusion matrices.

**Random Forest** emerged as the best model on the real-data notebook sample (F1 = 68.33%, ROC-AUC = 89.27%), confirming that ensemble tree methods effectively capture non-linear phishing patterns in URL structure. Feature importance analysis highlighted URL length, domain length, and digit count as the strongest predictors—consistent with known phishing obfuscation techniques.

The system extends beyond notebook experimentation through **Flask API deployment** and a polished **web dashboard** offering real-time scanning, batch analysis, model comparison, live analytics, and transparent rule-based risk scoring. Together, these components illustrate how classical machine learning—combined with thoughtful feature engineering and user-centered design—can be transformed into a practical tool for phishing awareness and URL vetting.

This project establishes a strong, reproducible foundation for more advanced fake URL detection work, including deep learning on character sequences, live blacklist integration, and browser extension deployment.

---

## FUTURE SCOPE

1. **Full-dataset training** — Train on all 549K URLs (or balanced subsample) with incremental learning or distributed processing.
2. **Advanced feature engineering** — Add WHOIS age, TLS certificate validity, domain entropy, typosquatting distance (Levenshtein to brand domains), and TLD reputation as model inputs.
3. **Hyperparameter optimization** — Systematic `GridSearchCV` / `RandomizedSearchCV` on Random Forest and XGBoost with F1-Score scoring.
4. **Class imbalance techniques** — Apply SMOTE or `class_weight='balanced'` to improve phishing recall.
5. **Deep learning models** — Character-level CNN/LSTM on raw URL strings for comparison with lexical features.
6. **SHAP explainability** — Add SHAP force plots for per-prediction feature attribution in the dashboard.
7. **Real-time blacklist feeds** — Integrate Google Safe Browsing, PhishTank, or OpenPhish APIs as an ensemble signal.
8. **Browser extension** — Lightweight client that calls the Flask API on link hover or before navigation.
9. **Model versioning & logging** — Persist scan logs, A/B test models, and monitor drift over time.
10. **Label consistency audit** — Ensure training label encoding (0=Safe, 1=Phishing) matches inference mapping in `app.py` for production reliability.

---

## REFERENCES

1. Ma, J., Saul, L. K., Savage, S., & Voelker, G. M. — *Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs* (Lexical URL features for malicious site detection).
2. Phishing Site URLs Dataset — Community-contributed labeled URL corpus (`phishing_site_urls.csv`).
3. Scikit-learn Documentation — Classification, `StandardScaler`, `cross_val_score`, metrics, pipelines. https://scikit-learn.org/
4. Flask Documentation — Web framework for API deployment. https://flask.palletsprojects.com/
5. XGBoost Documentation — Gradient boosting library. https://xgboost.readthedocs.io/
6. Chart.js Documentation — Frontend charting library. https://www.chartjs.org/
7. python-whois — Domain registration lookup library.
8. OWASP — Phishing Attack guidance and awareness resources. https://owasp.org/
9. Jain, A. K., & Gupta, B. B. — Survey papers on phishing detection using machine learning (ensemble methods, feature selection).
10. Symbiosis Institute of Technology — Supervised Machine Learning course materials.

---

## APPENDIX A — Project File Structure

```
SafeWeb-AI-Fake-URL-Detection-Using-Machine-Learning/
├── SafeWeb-code.ipynb          # Main ML pipeline (Phases 1–10)
├── phishing_site_urls.csv      # Primary dataset (549K URLs)
├── app.py                      # Flask API server
├── index.html                  # SafeWeb dashboard UI
├── train_all_models.py         # Model Lab multi-algorithm trainer
├── model_metrics.json          # Metrics for dashboard visualizations
├── requirements.txt            # Python dependencies
├── phishing_production_model.pkl   # Saved best model (after notebook run)
└── url_production_scaler.pkl       # Saved scaler (after notebook run)
```

## APPENDIX B — Core Feature Extraction Code

```python
def extract_features(url):
    try:
        if not url.startswith('http'):
            url = 'http://' + url
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return [
            len(url),
            len(domain),
            url.count('.'),
            url.count('-'),
            url.count('@'),
            url.count('/'),
            sum(c.isdigit() for c in url),
            1 if parsed_url.scheme == 'https' else 0,
            1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
        ]
    except:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## APPENDIX C — API Request Example

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure-login.fake-site.com/verify"}'
```

**Sample response fields:** `prediction`, `confidence`, `features`, `feature_names`, `risk.score`, `risk.level`, `risk.rules`

---

*Document generated for academic submission — SafeWeb AI Fake URL Detection Project*  
*Author: Ananya Srivastava | PRN: 24070722003*
