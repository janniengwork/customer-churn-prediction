# Credit Card Customer Churn Analysis

## Overview
End-to-end churn prediction on 10,127 bank credit card customers (BankChurners dataset, Kaggle): exploring who churns, building classifiers to flag at-risk customers, and identifying the behaviors that signal churn early.

## Methods
- Exploratory data analysis across demographics (age, gender, income, education) and account behavior
- Dropped two pre-computed Naive Bayes columns in the raw data to prevent target leakage
- Handled the 84/16 class imbalance with SMOTE, applied to the training set only
- Compared Logistic Regression (interpretable baseline) with Random Forest

## Results
| Model | ROC-AUC | Recall (churned) |
|---|---|---|
| Logistic Regression | 0.905 | 74% |
| Random Forest | 0.983 | 86% |

## Key Findings
- Churn is driven by behavior, not demographics. The top predictors are total transaction count, transaction amount, and revolving balance.
- Churned customers average about 45 transactions a year vs. 69 for retained customers, with 45% lower card utilization, so disengagement is visible before customers leave.
- Random Forest catches 86% of churners. In retention, recall matters most, since missing a churner costs more than a false alarm.

## How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
jupyter notebook churn_analysis.ipynb
```
The notebook reads `data/BankChurners.csv`.

## Stack
Python, Pandas, scikit-learn, imbalanced-learn, Matplotlib, Seaborn
