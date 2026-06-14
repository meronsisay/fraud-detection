# Fraud Detection System - E-commerce & Credit Card Transactions

## Project Overview

This project develops a unified fraud detection capability for Adey Innovations Inc., handling two distinct transaction streams:
- **E-commerce transactions** with rich user behavioral context
- **Bank credit card transactions** with anonymized PCA features

The system aims to balance the competing costs of false positives (customer friction) and false negatives (financial loss) through careful data preprocessing, feature engineering, and imbalanced learning techniques.

## Project Structure
```
fraud-detection/
├── .github/
│ └── workflows/
│ └── ci.yml # CI/CD pipeline
├── data/
│ ├── raw/ # Original datasets (gitignored)
│ └── processed/ # Cleaned feature-engineered data
├── notebooks/
│ ├── eda-fraud-data.ipynb # E-commerce EDA
│ ├── eda-creditcard.ipynb # Credit card EDA
│ └── modeling.ipynb # Model training & evaluation
├── src/
│ ├── init.py
│ ├── data_preprocessing.py # Data cleaning pipeline
│ ├── eda_utils.py # EDA helper functions
│ └── modeling.py # Model preparation and experimentation
├── tests/
│ ├── init.py
│ └── test_data_preprocessing.py # Unit tests for preprocessing
├── models/ # Saved model artifacts
├── requirements.txt # Python dependencies
└── README.md 
```

## Key EDA Insights

### E-Commerce Fraud Data

| Metric | Value |
|--------|-------|
| Total transactions | 151,112 |
| Fraud rate | 9.36% (1:9.7 ratio) |
| Time range | 2015-01-01 to 2015-12-16 |

**Top Fraud Indicators:**

| Feature | Correlation | Insight |
|---------|-------------|---------|
| `device_tx_velocity` | **+0.671** | Devices with 3+ transactions → 85% fraud rate |
| `time_since_signup` | **-0.258** | Fraud happens faster after signup |
| `age` | +0.007 | Minimal impact |
| `purchase_value` | +0.001 | No linear relationship |

**Geographic Pattern:** 
- Top 3 countries account for 60.9% of all transactions
- `'Unknown'` geolocation: 21,966 transactions (14.5%) with **8.57% fraud rate** (below 9.36% overall)
- Unknown IPs show LOWER fraud risk - likely legitimate corporate VPNs, cloud services
- **Decision:** Keep Unknown in dataset (not a fraud signal)

### Credit Card Fraud Data

| Metric | Value |
|--------|-------|
| Total transactions | 283,726 |
| Fraud rate | 0.167% (1:599 ratio) |
| Time span | 48 hours |

**Top Predictors (by correlation):**

| Feature | Correlation |
|---------|-------------|
| V14 | -0.293 |
| V12 | -0.251 |
| V10 | -0.207 |
| V3 | -0.182 |

**Time-Based Pattern:** 
- First 12 hours fraud rate: **0.35%**
- After 12 hours fraud rate: **0.14%**
- **2.5x higher risk in early transactions**

**Amount Pattern:** 
- Legitimate: Mean $88.41 | Median $22.00
- Fraud: Mean $123.87 | Median $9.82

## Feature Engineering

**E-commerce Features:**
- `time_since_signup` - Hours between signup and purchase (detects rapid fraud)
- `hour_of_day`, `day_of_week` - Temporal patterns
- `device_tx_velocity` - Transaction count per device (flags device abuse)
- `country` - IP geolocation (includes 'Unknown' for unmappable IPs - 14.5% of data, 8.57% fraud rate)
- `is_first_4hours` - Flags transactions within 4 hours of signup (3.3x higher risk window)
- `is_high_risk_country` - Flags high-risk nations (Ecuador, Tunisia, Peru) with ~26% fraud rate

## Class Imbalance Strategy

| Dataset | Original Ratio | Strategy | Target Ratio |
|---------|---------------|----------|--------------|
| E-commerce | 1:9.7 | SMOTE | 1:3 |
| Credit Card | 1:599 | SMOTE-ENN | 1:10 |

**Why SMOTE for E-commerce:** Moderate imbalance, rich feature space, prevents overfitting.

**Why SMOTE-ENN for Credit Card:** Extreme imbalance, PCA features benefit from noise removal.

## Data Transformation

| Step | Method | Rationale |
|------|--------|-----------|
| Scaling | StandardScaler | Zero mean, unit variance |
| Encoding | OneHotEncoder (drop='first') | Avoid multicollinearity |
| Split | Stratified 80/20 | Preserve class distribution |

## Modeling & Experiments

### Best Model Performance

| Dataset | Best Configuration | F1 Score | Precision | Recall | AUPRC |
|---------|-------------------|----------|-----------|--------|-------|
| **E-commerce** | XGBoost + SMOTE (1:5) | **0.6880** | 0.8535 | 0.5763 | 0.7068 |
| **Baseline** | Logistic Regression | 0.4208 | 0.2939 | 0.7403 | 0.6668 |
| **Credit Card** | XGBoost + SMOTE (1:20) | **0.8177** | 0.8605 | 0.7789 | 0.8081 |
| **Baseline** | Logistic Regression | 0.1059 | 0.0564 | 0.8737 | 0.7046 |

### Cross-Validation Stability

| Dataset | CV F1 Score | CV AUPRC |
|---------|-------------|----------|
| E-commerce | 0.6988 ± 0.0081 | 0.7219 ± 0.0059 |
| Credit Card | 0.8363 ± 0.0141 | 0.8393 ± 0.0281 |

### Key Experimental Findings

- **SMOTE outperformed undersampling** across both datasets, especially for extreme imbalance
- **Optimal ratios:** 1:5 for e-commerce, 1:20 for credit card
- **XGBoost > Random Forest** due to better handling of sparse features
- **Low cross-validation variance** confirms no overfitting
- **Dynamic hyperparameter tuning** improved F1 by 5-8% over default configurations

### Confusion Matrix Results

**E-commerce Best Model (SMOTE 0.2):**

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| Actual Legit | 26,938 | 1,412 |
| Actual Fraud | 263 | 359 |

**Credit Card Best Model (SMOTE 0.05):**

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| Actual Legit | 56,597 | 279 |
| Actual Fraud | 14 | 49 |



## Environment Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/meronsisay/fraud-detection.git
cd fraud-detection

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

jupyter notebook notebooks/eda-fraud-data.ipynb
jupyter notebook notebooks/eda-creditcard.ipynb
jupyter notebook notebooks/modeling.ipynb