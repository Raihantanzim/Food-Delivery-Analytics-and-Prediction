# Food Delivery Analytics & Prediction
![image alt](https://github.com/Raihantanzim/Machine-Learning/blob/9bea35245988ff7fef6f43f2cebe7c19c45cdc96/Machine%20Learning%20in%20Food%20Delivery%20Operations.jpg)

Machine Learning coursework on a food delivery dataset (1,000 orders): EDA + K-Means clustering, delivery-time regression (Random Forest), and complaint prediction (Balanced Random Forest) using creation-time features only to avoid leakage.


Food Delivery ML Coursework (COMP1816)

This repository contains my COMP1816 Machine Learning coursework using a historical food delivery dataset (1,000 orders; 18 columns). The work follows a leakage-safe setup by using only order creation-time information for prediction.

Project goals

Exploratory Data Analysis (EDA): summarize distributions, missingness, and relationships.

Clustering: segment orders using K-Means (selection guided by elbow + silhouette).

Regression: predict delivery time in minutes at order creation.

Classification: predict whether an order will generate a complaint (imbalanced classification).

Key results (holdout test)

Delivery-time regression (Random Forest): MAE 12.57 mins, RMSE 17.05, R² 0.208.
Complaint prediction (Balanced Random Forest): ROC-AUC 0.692, PR-AUC 0.482, F1 0.452 (threshold 0.5).
Clustering (K-Means): best silhouette at k=2 but weak separation (silhouette ≈ 0.148).

Methods used

Preprocessing: imputation + one-hot encoding; engineered creation-time features (workload ratios, basket stats, cyclic time-of-day/day-of-week encodings).

Regression models: Dummy mean baseline, Ridge, Random Forest (tuned).

Classification models: Dummy (most frequent), Logistic Regression (balanced), Random Forest (balanced), Balanced Random Forest (tuned).

Evaluation: MAE/RMSE/R² for regression; ROC-AUC/PR-AUC/F1 + confusion matrix for classification.


How to run

Install dependencies (Python 3.9+ recommended).

Open and run the notebook(s) in notebooks/ to reproduce figures and results.

The final write-up is in report/.

Notes

Complaint prediction is imbalanced (~24.4% positives), so PR-AUC and threshold choice matter operationally.

Performance is bounded by missing real-world drivers (distance, traffic, prep time), which are not in the dataset.
