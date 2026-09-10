# CreditWise — Loan Approval Prediction

An end-to-end supervised machine learning pipeline that predicts whether a loan application will be approved, based on applicant financial and demographic data. Built as a hands-on project to practice the full ML workflow: EDA, feature engineering, model comparison, and evaluation.

## Overview

- **Task:** Binary classification (Loan Approved: Yes / No)
- **Models compared:** Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes
- **Best model:** Naive Bayes — 80.4% precision, 86.5% accuracy
- **Dataset:** ~1,000 applicant records with 19 features

## Dataset

Each row represents a loan applicant, with features including:

- **Financial:** Applicant Income, Coapplicant Income, Credit Score, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term, Existing Loans
- **Demographic:** Age, Gender, Marital Status, Dependents, Education Level
- **Categorical:** Employment Status, Employer Category, Property Area, Loan Purpose
- **Target:** `Loan_Approved` (Yes / No)

## Workflow

1. **Exploratory Data Analysis (EDA)** — distributions, missing values, correlation heatmap between numerical features
2. **Preprocessing** — train/test split (80/20), feature scaling with `StandardScaler`
3. **Feature Engineering** — squared terms for Credit Score and DTI Ratio, log-transform on Applicant Income
4. **Model Training** — Logistic Regression, KNN (k=5), Gaussian Naive Bayes
5. **Evaluation** — Precision, Recall, F1-score, Accuracy, Confusion Matrix

## Results

| Model               | Precision | Recall | F1-score | Accuracy  |
| ------------------- | --------- | ------ | -------- | --------- |
| Logistic Regression | 0.783     | 0.770  | 0.777    | 0.865     |
| KNN (k=5)           | 0.627     | 0.525  | 0.571    | 0.760     |
| **Naive Bayes**     | **0.804** | 0.738  | 0.769    | **0.865** |

Naive Bayes was selected as the best-performing model based on precision, with Logistic Regression a close second on F1-score. KNN underperformed both, likely due to sensitivity to feature scaling and the curse of dimensionality with 19 features.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

## Project Structure

```
credit_wise.ipynb   # Full notebook: EDA, preprocessing, modeling, evaluation
```

## Key Takeaways

- Compared three fundamentally different classifiers (linear, instance-based, probabilistic) on the same dataset
- Practiced evaluating models beyond accuracy alone, since precision/recall trade-offs matter for loan approval decisions (false approvals vs. false rejections carry different costs)
- Applied feature engineering techniques (polynomial features, log transforms) to handle skewed distributions
