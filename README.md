# IEEE-CIS E-commerce Fraud Detection System|
###Link
Streamlit link:https://e-commercefrauddetectionsystem-qrnevqb3ixftranizkitjf.streamlit.app/

End-to-End Fraud Detection Pipeline + Real-Time Risk Scoring Application

------------------------------------------------------------------------

# Project Overview

This project implements a production-style fraud detection system built
using the Kaggle IEEE-CIS Fraud Detection dataset.

The goal is to demonstrate a full machine learning pipeline including:

• Data ingestion\
• Feature engineering\
• Model training\
• Risk scoring\
• Operational risk banding\
• Real-time scoring API\
• Interactive UI for fraud analysis

The project mimics real-world fraud detection systems used by payment
processors and e-commerce companies.

------------------------------------------------------------------------

# Technologies Used

Language: Python\
ML Model: XGBoost\
Data Processing: Pandas, NumPy\
API Framework: FastAPI\
User Interface: Streamlit\
Storage Format: Parquet\
Serialization: Joblib

------------------------------------------------------------------------

# Dataset

Dataset: IEEE-CIS Fraud Detection (Kaggle)

Approximate dataset properties:

Records: \~590,000\
Fraud rate: \~3--4%\
Total features: \~430+ engineered variables

Goal: Predict whether a transaction is fraudulent.

------------------------------------------------------------------------

# Project Architecture

Dataset → Feature Engineering → Model Training → Risk Scoring → API → UI

The system supports both batch scoring and real-time inference.

------------------------------------------------------------------------

# Project Structure

ieee_fraud_major_project/

app/\
streamlit_app.py → Streamlit user interface

src/\
api.py → FastAPI scoring service\
build_dataset.py → dataset construction\
config.py → configuration paths\
features.py → feature engineering\
make_parquet.py → CSV to Parquet conversion\
make_submission.py → batch scoring + Kaggle submission\
train_xgb_time_split.py → model training\
tune_thresholds.py → threshold tuning\
report_plots.py → evaluation charts\
utils.py → helper utilities\
db.py → API scoring history

data/\
raw → original dataset\
interim → intermediate outputs\
processed → feature-engineered dataset

models/\
xgb_model.joblib

reports/\
metrics.json\
thresholds.json\
submission.csv\
test_scored.csv

requirements.txt

------------------------------------------------------------------------

# Environment Setup

Step 1 --- Navigate to Project

cd
"C:`\Users`{=tex}`\HP`{=tex}`\OneDrive`{=tex}`\Desktop`{=tex}`\Major `{=tex}project"

Step 2 --- Create Virtual Environment

python -m venv .venv

Step 3 --- Activate

..venv`\Scripts`{=tex}`\Activate`{=tex}.ps1

Step 4 --- Install Dependencies

python -m pip install --upgrade pip pip install -r requirements.txt

------------------------------------------------------------------------

# Add Kaggle Dataset

Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection

Place zip file in project root:

ieee-fraud-detection.zip

Extract:

Expand-Archive -Path ".`\ieee`{=tex}-fraud-detection.zip"
-DestinationPath ".`\data`{=tex}`\raw`{=tex}" -Force

Expected files:

train_transaction.csv\
train_identity.csv\
test_transaction.csv\
test_identity.csv\
sample_submission.csv

------------------------------------------------------------------------

# Run Training Pipeline

Run these commands in order:

python -m src.make_parquet

python -m src.build_dataset

python -m src.train_xgb_time_split

python -m src.tune_thresholds

python -m src.make_submission

------------------------------------------------------------------------

# Pipeline Outputs

Model:

models/xgb_model.joblib

Reports:

reports/metrics.json\
reports/thresholds.json\
reports/submission.csv

------------------------------------------------------------------------

# Risk Score Logic

risk_score = round(100 × fraud_probability)

Risk Bands:

HIGH → manual review required

MEDIUM → additional verification

LOW → automatically approved

Thresholds are determined using validation data distribution.

------------------------------------------------------------------------

# Run Real-Time API

uvicorn src.api:app --reload

Health endpoint:

http://127.0.0.1:8000/health

------------------------------------------------------------------------

# Launch Streamlit Interface

streamlit run app/streamlit_app.py

The interface allows:

• manual transaction scoring\
• batch CSV scoring\
• fraud risk visualization\
• scoring history tracking

------------------------------------------------------------------------

# Why Two Risk Signals Exist

The IEEE dataset contains more than 400 engineered features.

A real user cannot manually input that many attributes.

Therefore the system uses two scoring layers:

1.  Machine Learning Model Uses full feature space and produces accurate
    probability predictions.

2.  Interactive Risk Engine Uses a small set of user-enterable inputs to
    generate interpretable scores for demonstration purposes.

This hybrid design reflects real production fraud systems.

------------------------------------------------------------------------

# Input Fields Explained

Transaction Amount\
Represents purchase value. Higher values often increase fraud risk.

Product Category\
Certain items like electronics or gift cards have higher fraud rates.

Card Type\
Prepaid cards are commonly used in fraud.

Card Brand\
Helps detect brand-specific fraud patterns.

Billing Country\
Fraud risk increases when billing location differs from transaction
context.

Shipping Country\
Cross-border shipping can increase fraud likelihood.

Billing vs Shipping Match\
Mismatch is a strong fraud indicator.

Distance Between Billing and Shipping\
Large geographic distance often signals suspicious activity.

Email Domain\
Disposable email services increase risk.

Device Type\
Fraud rings sometimes use automated mobile environments.

Browser\
Anonymity browsers may indicate higher risk.

------------------------------------------------------------------------

# Example Output

Risk Score: 82\
Risk Band: HIGH

Reasons: High transaction value\
Billing and shipping mismatch\
Large geographic distance\
High resale product category

------------------------------------------------------------------------

# Model Evaluation

Typical validation metrics:

PR-AUC: \~0.56\
ROC-AUC: \~0.91\
Fraud Rate: \~3.4%\
Feature Count: \~445

------------------------------------------------------------------------

# Testing the System

Manual Testing

Use Streamlit interface and input transaction values.

Batch Testing

Upload CSV file or score dataset rows.

------------------------------------------------------------------------

# Important Design Choices

Time-based train/validation split to reduce leakage

PR-AUC optimization for class imbalance

scale_pos_weight tuning for fraud imbalance

Capacity-based risk thresholding

Hybrid ML + rule-based scoring

FastAPI + Streamlit architecture

Parquet-based data pipeline

------------------------------------------------------------------------

# Possible Future Improvements

Graph-based fraud networks

Velocity features

Autoencoder anomaly detection

SHAP explainability dashboard

Online learning simulation

Feature store integration

------------------------------------------------------------------------

# Project Status

Data pipeline: complete

Model training: complete

API service: working

Streamlit interface: working

Batch scoring: enabled
