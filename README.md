# Food Delivery Analytics & Prediction
![image alt](https://github.com/Raihantanzim/Machine-Learning/blob/9bea35245988ff7fef6f43f2cebe7c19c45cdc96/Machine%20Learning%20in%20Food%20Delivery%20Operations.jpg)

## Overview
This project analyses a historical food delivery dataset to support operational decision-making in three main areas:

1. **Exploratory Data Analysis (EDA)** to understand order patterns and data quality  
2. **Delivery Time Prediction** at the moment an order is placed  
3. **Complaint Prediction** using only information available at order creation time  

A key design principle of this project is **leakage-safe modelling**: only creation-time features are used for prediction, ensuring the models reflect realistic operational use. :contentReference[oaicite:1]{index=1}

---

## Problem Statement
Food delivery platforms need to make accurate, early decisions about:

- how long an order is likely to take
- which orders are at risk of customer complaints
- how to prioritise operational interventions during busy periods

This project explores whether delivery delays and complaint risk can be predicted from order, basket, temporal, and workload-related features available **when the order is created**. :contentReference[oaicite:2]{index=2}

---

## Project Objectives
The main objectives of this project are to:

- analyse the structure, missingness, and distributions in the dataset
- identify broad order segments using clustering
- predict **delivery time** as a regression task
- predict **complaint occurrence** as an imbalanced classification task
- compare baseline and tuned machine learning models
- interpret the operational usefulness of the results :contentReference[oaicite:3]{index=3}

---

## Dataset
The dataset contains:

- **1,000 orders**
- **18 columns**
- **24.4% complaint rate**
- average delivery time of **47.27 minutes**
- median delivery time of **44.27 minutes**
- delivery times ranging from **16.12 to 133.43 minutes** :contentReference[oaicite:4]{index=4}

### Features include:
- market / area
- store category
- order protocol
- total items
- subtotal
- min and max item price
- total on-shift partners
- total busy partners
- total outstanding orders
- created day / holiday / timestamp

### Targets created:
- **delivery_mins**: time from order creation to actual delivery
- **complaint_flag**: binary complaint indicator derived from complaint records :contentReference[oaicite:5]{index=5}

---

## Key Insights from EDA
The exploratory analysis showed that:

- delivery times are **right-skewed**, with occasional extreme delays
- complaint prediction is an **imbalanced classification problem**
- complaint-related fields have the highest missingness (**75.6%**)
- workload variables have modest missingness (around **8%**)
- operational congestion and workload features are likely to influence both delays and complaints :contentReference[oaicite:6]{index=6}

---

## Clustering
K-Means clustering was used to segment orders based on preprocessed creation-time features.

### Clustering Result
- best number of clusters: **k = 2**
- silhouette score: **0.148**

This suggests that order segments exist, but the separation is weak, so the clusters are best interpreted as **broad operational segments** rather than strongly distinct order types. :contentReference[oaicite:7]{index=7}

---

## Methodology

### 1. Preprocessing
- median imputation for numeric missing values
- most-frequent imputation for categorical missing values
- one-hot encoding for categorical features
- scaling where needed for consistent pipelines

### 2. Feature Engineering
Features were engineered using only creation-time information, including:

- **avg_item_price**
- **price_range**
- **busy_ratio**
- **free_partners**
- **outstanding_per_partner**
- cyclical encodings for **time of day**
- cyclical encodings for **day of week** :contentReference[oaicite:8]{index=8}

### 3. Leakage Prevention
The following variables were excluded from prediction models:

- `actual_delivery_time`
- `delivery_mins` (as predictor)
- complaint-related label fields

This ensured a realistic prediction setting. :contentReference[oaicite:9]{index=9}

---

## Models Used

### Regression Task: Delivery Time Prediction
Models compared:
- Dummy Mean Regressor
- Ridge Regression
- **Random Forest Regressor** (main model)

### Classification Task: Complaint Prediction
Models compared:
- Dummy Most Frequent
- Logistic Regression (balanced)
- Random Forest (balanced)
- **Balanced Random Forest** (main model) :contentReference[oaicite:10]{index=10}

---

## Results

### Delivery Time Prediction
The tuned **Random Forest Regressor** achieved on the holdout test set:

- **MAE = 12.57 minutes**
- **RMSE = 17.05 minutes**
- **R² = 0.208** :contentReference[oaicite:11]{index=11}

This outperformed both the dummy baseline and Ridge regression, showing that creation-time features contain useful predictive signal, although performance is limited by unobserved factors such as traffic, travel distance, and preparation-time variability. :contentReference[oaicite:12]{index=12}

### Complaint Prediction
The tuned **Balanced Random Forest** achieved on the holdout test set:

- **ROC-AUC = 0.692**
- **PR-AUC = 0.482**
- **F1-score = 0.487**
- **Precision = 0.359**
- **Recall = 0.755**  
  at threshold **0.43** :contentReference[oaicite:13]{index=13}

These results show that the model is useful for **risk ranking** and identifying likely complaint cases, especially when recall is important.

---

## Operational Value
This project has practical value for delivery platforms because it can support:

- **ETA estimation** at order creation
- identifying **high-risk delay orders**
- proactive customer messaging
- dispatch prioritisation
- customer support intervention for complaint-risk orders
- staffing and workload planning during busy periods :contentReference[oaicite:14]{index=14}

---

## Tech Stack
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
