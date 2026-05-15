# =========================
# CREDIT RISK MODEL (CLEAN FIXED VERSION)
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE
import joblib

# =========================
# PATH SETUP (IMPORTANT FIX)
# =========================

BASE_DIR = r"C:\Users\Admin\Desktop\AI_Credit_Risk_Project"

DATA_PATH = os.path.join(BASE_DIR, "Data", "Delinquency_prediction_dataset.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output", "Reports")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
VIS_DIR = os.path.join(BASE_DIR, "Visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_excel(DATA_PATH)

print("\n===== DATA LOADED =====")
print(df.shape)
print(df.head())

# =========================
# 2. CLEAN DATA
# =========================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# =========================
# 3. ENCODING
# =========================

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\n===== ENCODING DONE =====")

# =========================
# 4. SPLIT DATA
# =========================

target = "Delinquent_Account"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. SMOTE (BALANCE DATA)
# =========================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train.value_counts())

# =========================
# 6. SCALING (FIXED IMPORT ERROR)
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# =========================
# 7. MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

print("\n===== MODEL TRAINED =====")

# =========================
# 8. PREDICTION
# =========================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 9. SAVE OUTPUT EXCEL
# =========================

output = pd.DataFrame(X_test, columns=X.columns)
output["Actual"] = y_test.values
output["Predicted"] = y_pred

output_file = os.path.join(OUTPUT_DIR, "delinquency_predictions_output.xlsx")
output.to_excel(output_file, index=False)

print("\nOUTPUT SAVED:", output_file)

# =========================
# 10. VISUALS (FIXED PATHS)
# =========================

# 1. Class Distribution
plt.figure()
sns.countplot(x=y)
plt.title("Original Class Distribution")
plt.savefig(os.path.join(VIS_DIR, "class_distribution.png"))
plt.show()

# 2. SMOTE Distribution
plt.figure()
sns.countplot(x=y_train)
plt.title("After SMOTE Distribution")
plt.savefig(os.path.join(VIS_DIR, "smote_distribution.png"))
plt.show()

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"))
plt.show()

# 4. Feature Importance
plt.figure()
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.savefig(os.path.join(VIS_DIR, "feature_importance.png"))
plt.show()

print("\n===== ALL VISUALS SAVED =====")