# 💳 Credit Risk / Delinquency Prediction System

A Machine Learning project that predicts customer loan delinquency using financial and behavioral data. The model is trained using classification algorithms and handles class imbalance using SMOTE.

---

## 📌 Project Objective

To build a predictive system that can:
- Identify customers likely to default (delinquent accounts)
- Help financial institutions reduce credit risk
- Improve decision-making using data-driven insights

---

## 📊 Dataset Description

The dataset contains customer financial and repayment behavior features such as:

- Age
- Income
- Credit Score
- Loan Amount
- Repayment behavior (Month_1 to Month_6)
- Target variable: **Delinquent_Account (0 = No, 1 = Yes)**

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Jupyter/Spyder
- Joblib (model saving)

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- Missing value handling (mean/mode imputation)
- Label Encoding for categorical variables

### 2. Data Splitting
- Train-Test Split (80/20)

### 3. Handling Imbalanced Data
- SMOTE (Synthetic Minority Oversampling Technique)

### 4. Feature Scaling
- StandardScaler applied to improve model performance

### 5. Model Training
- Random Forest Classifier (primary model)
- Balanced class weights used

---

## 📈 Model Performance

- Accuracy: ~70% (varies based on run)
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📊 Visualizations

The project generates the following insights:

- Class Distribution (Before & After SMOTE)
- Confusion Matrix
- Feature Importance (Risk Drivers)

All visuals are saved in: