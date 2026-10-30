# Robust Network Intrusion Detection: Multiclass Traffic Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green.svg)](https://scikit-learn.org/)
[![Metric](https://img.shields.io/badge/Macro--F1-0.9909-brightgreen.svg)]()

A high-performance, generalized machine learning pipeline for multiclass network traffic intrusion detection. This solution is specifically engineered to handle **traffic behavior distribution shifts**, **severe class imbalance**, and **out-of-distribution network flows**, evaluated using the **Macro-Averaged F1 Score**.

---

## 📌 Challenge Overview

Traditional intrusion detection systems (IDS) excel on familiar traffic patterns but degrade sharply when encountering unfamiliar traffic behaviors. In this competition, the evaluation data contains reserved network traffic profiles deliberately separated from the training distribution.

- **Objective**: Accurately classify network traffic flows into 4 distinct categories.
- **Evaluation Metric**: **Macro-Averaged F1 Score** across all four classes:
  $$\text{Macro } F_1 = \frac{1}{4} \sum_{c=1}^{4} F_{1, c}$$
- **Target Categories**:
  1. `Benign`: Normal, non-malicious user and background traffic.
  2. `Availability`: Resource-exhaustion and Denial-of-Service (DoS/DDoS) flood attacks.
  3. `Reconnaissance`: Port scans, host scans, and vulnerability probes.
  4. `Spoofing`: IP, MAC, or ARP header manipulation and identity forgery.

---

## 🔍 Dataset Architecture & Insights

| Dataset Partition | Record Count | Features | Description |
| :--- | :---: | :---: | :--- |
| `train.csv` | 84,713 | 84 | Labeled traffic flows with `id`, 82 network features, and `Target` |
| `test.csv` | 33,161 | 83 | Unlabeled test flows with `id` and 82 network features |
| `sample_submission.csv` | 33,161 | 2 | Required format (`id`, `Target`) |

### Key Findings from Exploratory Data Analysis (EDA)

1. **Extreme Class Imbalance in Training Data**:
   - `Availability`: 80,072 flows (**94.52%**)
   - `Spoofing`: 2,323 flows (**2.74%**)
   - `Benign`: 1,583 flows (**1.87%**)
   - `Reconnaissance`: 735 flows (**0.87%**)
   - *Impact*: Standard models optimize accuracy and predict the majority class, achieving ~94% accuracy but a disastrous Macro F1 score of ~0.24.

2. **Severe Port & Behavior Distribution Shift**:
   - In `train.csv`, **94%** of flows target destination port `21` (FTP flood).
   - In `test.csv`, **1,501** destination ports were never seen in training (e.g. DNS port `53`, mDNS `5353`, Web `80/443`).
   - *Strategy*: Models must **not** memorize raw port numbers. Instead, we extract port types (well-known, registered, dynamic) and focus on intrinsic network flow dynamics (IAT, packet symmetry, TCP flag interactions, window sizes).

---

## 🛠️ Architecture & Engineering Strategy

### 1. Domain-Specific Feature Engineering (`src/features.py`)
- **Port Ranges**: IANA classification (Well-Known `<1024`, Registered `1024-49151`, Ephemeral/Dynamic `≥49152`).
- **Flow Symmetry Ratios**: Forward-to-backward packet ratio and payload ratio.
- **Header-to-Payload Dynamics**: Header size vs payload ratio (essential for detecting header-heavy DoS floods and reconnaissance scans).
- **TCP Flag Balances**: SYN-to-ACK ratio, RST-to-packet ratio, FIN flag frequency.
- **TCP Window Dynamics**: Initial vs last window size discrepancies and fwd/bwd window size delta.
- **Inter-Arrival Time (IAT) Spreads**: Duration spreads and jitter across forward/backward flows.
- **Skew Mitigation**: `log1p` transformation on heavy-tailed traffic rate metrics.

### 2. Class Imbalance Optimization
- Inversely proportional balanced sample weights calculated per class:
  $$\text{weight}_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
- Gives minority classes (`Reconnaissance` @ 0.87% and `Benign` @ 1.87%) equal loss contribution during gradient boosting optimization.

### 3. Cross-Validation & Modeling (`src/train.py`)
- **5-Fold Stratified K-Fold Cross-Validation** to ensure all folds preserve the exact minority class proportions.
- **XGBoost Histogram Trees (`tree_method='hist'`)** with regularized depth, column subsampling (`colsample_bytree=0.85`), and row subsampling (`subsample=0.85`) to prevent overfitting.
- **Fold Ensembling**: Final test predictions average the predicted probability distributions across all 5 trained fold models.

---

## 📊 Experimental Results

### 5-Fold Stratified Cross-Validation Scores

| Fold | Macro F1 Score |
| :---: | :---: |
| **Fold 1** | `0.9899` |
| **Fold 2** | `0.9931` |
| **Fold 3** | `0.9902` |
| **Fold 4** | `0.9934` |
| **Fold 5** | `0.9880` |
| **Mean OOF Macro F1** | **`0.9909 ± 0.0021`** |

### Out-of-Fold Classification Report

| Target Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 0.9836 | 0.9848 | **0.9842** | 1,583 |
| **Availability** | 1.0000 | 1.0000 | **1.0000** | 80,072 |
| **Reconnaissance** | 0.9905 | 0.9905 | **0.9905** | 735 |
| **Spoofing** | 0.9892 | 0.9888 | **0.9890** | 2,323 |
| **Macro Average** | **0.9908** | **0.9910** | **0.9909** | **84,713** |
| **Overall Accuracy** | — | — | **0.9993** | **84,713** |

---

## 📁 Repository Structure

```text
network_intrusion_detection/
├── .gitignore                      # Git ignore rules for data science
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Pinned project dependencies
│
├── data/                           # Dataset storage
│   ├── README.md                   # Dataset details and column descriptions
│   ├── train.csv                   # Labeled training data (84,713 rows)
│   ├── test.csv                    # Unlabeled test data (33,161 rows)
│   └── sample_submission.csv       # Required format template
│
├── notebooks/                      # Interactive Jupyter Notebooks
│   ├── 01_exploratory_data_analysis.ipynb   # Complete visual EDA
│   └── 02_end_to_end_pipeline.ipynb         # Self-contained reproducible pipeline
│
├── src/                            # Modular production source code
│   ├── __init__.py
│   ├── config.py                   # Paths, parameters, and class label mappings
│   ├── preprocess.py               # Data cleaning & missing/infinite value handler
│   ├── features.py                 # Network flow domain feature engineering
│   ├── train.py                    # 5-fold stratified training & ensembling
│   ├── predict.py                  # Standalone inference & submission generator
│   ├── eda.py                      # Automated dataset summary and statistics
│   └── evaluate.py                 # Out-of-fold evaluation & confusion matrix
│
├── models/                         # Serialized model artifacts
│   ├── feature_columns.json        # 127 engineered feature schema
│   ├── label_mapping.json          # Target-to-index mappings
│   ├── oof_predictions.csv         # Out-of-fold validation predictions
│   ├── xgb_fold_0.json             # Trained fold 1 booster
│   ├── xgb_fold_1.json             # Trained fold 2 booster
│   ├── xgb_fold_2.json             # Trained fold 3 booster
│   ├── xgb_fold_3.json             # Trained fold 4 booster
│   └── xgb_fold_4.json             # Trained fold 5 booster
│
└── submissions/                    # Competition submission output
    ├── sample_submission.csv       # Baseline benchmark
    └── submission.csv              # Final generated predictions (33,161 rows)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository (or navigate to project folder)
cd network_intrusion_detection

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis
```bash
python src/eda.py
```

### 3. Train Models & Generate Submission
```bash
python src/train.py
```
This runs 5-Fold Stratified cross-validation, saves the trained models to `models/`, and generates `submissions/submission.csv`.

### 4. Run Standalone Inference (Optional)
```bash
python src/predict.py
```

### 5. Evaluate Out-of-Fold Results
```bash
python src/evaluate.py
```

---

## 📤 Step-by-Step Instructions to Push to GitHub

Follow these steps in your terminal or PowerShell to push this project to your GitHub account:

### Step 1: Initialize Git Repository
```bash
cd C:\Users\CVR\.gemini\antigravity\scratch\network_intrusion_detection
git init
```

### Step 2: Configure Your Git Identity (If not already set)
```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

### Step 3: Stage and Commit the Files
```bash
git add .
git commit -m "feat: complete robust network intrusion detection pipeline with 0.9909 Macro-F1"
```

> **Note on Large CSVs**: `train.csv` is ~42MB, which is below GitHub's 100MB limit. If you prefer to push code without the 42MB training CSV, uncomment `data/*.csv` in `.gitignore` before running `git add`.

### Step 4: Create a New Repository on GitHub
1. Open your browser and go to [github.com/new](https://github.com/new).
2. Set **Repository name** to: `network-intrusion-detection` (or your preferred name).
3. Set visibility to **Public** or **Private**.
4. Leave **Initialize this repository with a README** unchecked (we already have a complete README).
5. Click **Create repository**.

### Step 5: Link and Push to GitHub
Copy your repository URL from GitHub and run:
```bash
# Rename branch to main
git branch -M main

# Add your GitHub remote (replace with your actual GitHub username and repo name)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/network-intrusion-detection.git

# Push code to GitHub
git push -u origin main
```

---

## 🏆 Submission Verification

The generated `submissions/submission.csv` has been verified against competition guidelines:
- **Exact Row Count**: 33,161 records (matching `test.csv` IDs exactly).
- **Exact Column Headers**: `id,Target`.
- **Target Classes**: All predicted values belong strictly to `{'Benign', 'Availability', 'Reconnaissance', 'Spoofing'}`.
- **Zero Nulls / Missing Values**: 100% complete predictions.
