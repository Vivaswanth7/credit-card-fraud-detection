Credit Card Fraud Detection
📌 Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions.
It handles severe class imbalance using SMOTE and trains a Random Forest classifier to achieve high fraud detection performance.

📊 Dataset

Source: Kaggle – Credit Card Fraud Detection

Transactions: 284,807

Fraud cases: 492 (0.17%)

Features: 30 anonymized PCA components + Time + Amount

Target: Class → 0 = Legitimate, 1 = Fraud

⚙️ Workflow

EDA (Exploratory Data Analysis) → Class imbalance, distributions of Amount & Time.

Preprocessing → Standard scaling with StandardScaler.

Balancing → SMOTE oversampling to create balanced fraud/non-fraud dataset.

Model Training → Random Forest Classifier.

Evaluation → Classification Report, Confusion Matrix, ROC AUC, ROC Curve.

📈 Results

On imbalanced dataset → high accuracy but poor recall for fraud (missed fraud cases).

On SMOTE-balanced dataset →

Precision (Fraud): ~0.9999

Recall (Fraud): ~1.0

ROC AUC: ~1.0

✅ Model successfully detects fraud cases with near-perfect performance on test set.

🚀 How to Run

Clone this repo or open notebook in Google Colab.

Install dependencies:

pip install imbalanced-learn scikit-learn pandas matplotlib seaborn joblib


Download dataset from Kaggle and place creditcard.csv in working directory.

Run notebook cells → results will be displayed (classification report, confusion matrix, ROC curve).

📷 Visualizations

Confusion Matrix

ROC Curve

(Add screenshots from your Colab outputs here)

📌 Key Learnings

Handling class imbalance is critical in fraud detection.

SMOTE dramatically improves fraud recall compared to raw imbalanced training.

Random Forest gives a strong baseline, achieving near-perfect fraud classification.

✍️ Developed by Vivaswanth Sadasivuni
