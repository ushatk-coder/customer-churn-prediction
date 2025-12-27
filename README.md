
# Customer Churn Prediction Project

## Overview
This project predicts customer churn (Yes/No) using the Telco Customer Churn dataset.
The final model is a Logistic Regression classifier with balanced class weights to handle class imbalance.

## Files
- churn_model_final.py : Standalone Python script for training and evaluating the final model
- WA_Fn-UseC_-Telco-Customer-Churn.csv : Dataset (download separately from Kaggle)

## Model Used
- Logistic Regression
- class_weight = 'balanced'

## Why This Model?
- Maximizes recall for churn customers
- Reduces missed churners
- Interpretable and business-friendly

## How to Run

### 1. Install dependencies
pip install pandas scikit-learn

### 2. Run the model
python churn_model_final.py

## Output
- Accuracy
- Confusion Matrix
- Classification Report

## Author
Usha
