# Project 3: Machine Learning System for Fraud Detection
### RISE Internship — Machine Learning & AI

---

## Problem Statement
Financial institutions and e-commerce platforms face increasing fraudulent transactions resulting in major financial losses. Rule-based systems fail to detect evolving fraud patterns. This project builds an ML model that identifies suspicious transactions in real time.

## Objective
Build a machine learning model that detects fraudulent transactions based on transaction behavior and historical data.

---

## Project Structure
```
fraud_detection/
├── fraud_detection.py          ← Main script (all steps end-to-end)
├── README.md                   ← This file
└── outputs/
    ├── 01_eda_overview.png             ← Class balance, amount dist, fraud by hour
    ├── 02_feature_distributions.png   ← Feature distributions by class
    ├── 03_model_comparison.png        ← Precision, Recall, F1, AUC bar chart
    ├── 04_confusion_matrices.png      ← Confusion matrix for all 4 models
    ├── 05_roc_curves.png              ← ROC curves comparison
    ├── 06_precision_recall_curve.png  ← P-R curve for best model
    ├── 07_feature_importance.png      ← Top 15 features (Random Forest)
    ├── 08_fraud_patterns_correlation.png ← Fraud vs Legit correlation heatmaps
    └── suspicious_transactions.csv   ← Flagged high-risk transactions
```

---

## Tools & Libraries
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Plotting & visualization |
| Seaborn | Statistical visualizations |
| Scikit-learn | ML models & evaluation |
| Jupyter Notebook | Interactive development |

---

## Pipeline Steps

### Step 1 — Dataset Ingestion
- Synthetic dataset of 10,000 transactions (9,500 legit + 500 fraud)
- Features: Time, Amount, V1–V5 (behavioral features), Hour, MerchantCategory
- Realistic 5% fraud rate to simulate class imbalance

### Step 2 — EDA (Exploratory Data Analysis)
- Class distribution visualization
- Transaction amount comparison (fraud vs legit)
- Fraud frequency by hour of day
- Feature distribution overlays

### Step 3 — Data Cleaning & Preprocessing
- Zero missing values verified
- One-hot encoding for categorical feature (MerchantCategory)
- StandardScaler normalization on Amount and Time

### Step 4 — Handling Class Imbalance
- Train/test split (80/20) with stratification — BEFORE resampling
- Oversampling (resample) applied only to training data
- Balanced 50/50 split for training

### Step 5 — Feature Engineering
- V1×V2 interaction feature
- V4×V5 interaction feature
- Amount/V1 ratio
- V-feature sum and standard deviation

### Step 6 — Model Training
Four classifiers trained and compared:
- Logistic Regression
- Decision Tree (max_depth=8)
- Random Forest (100 estimators)
- Gradient Boosting (100 estimators)

### Step 7 — Model Evaluation
Metrics used: **Precision, Recall, F1-Score, ROC-AUC**

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.840 | 1.000 | 0.913 | 1.000 |
| Decision Tree | 0.825 | 0.990 | 0.900 | 0.991 |
| Random Forest | 1.000 | 0.950 | 0.974 | 1.000 |
| **Gradient Boosting** | **0.952** | **1.000** | **0.976** | **1.000** |

**Best Model: Gradient Boosting (F1 = 0.9756)**

### Step 8 — Fraud Pattern Visualization
- Correlation heatmaps (fraud vs legit)
- ROC curves for all models
- Precision-Recall curve
- Feature importance chart

### Step 9 — Suspicious Transaction Prediction
- Fraud probability score generated for all test transactions
- Transactions with probability > 0.70 flagged as suspicious
- Exported to CSV for business action

---

## Key Results
- **99 out of 99** suspicious-flagged transactions were confirmed fraud
- Gradient Boosting achieved **perfect recall** — no fraud transactions missed
- Top fraud indicators: V2 (high positive), V1 (high negative), V5 (high negative)

---

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python fraud_detection.py
```

---

## Job Roles Targeted
- Machine Learning Engineer
- Fraud Analyst
- Data Scientist

---

*RISE Internship | Tamizan Skills | www.tamizhanskills.com*
