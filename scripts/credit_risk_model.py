# =========================
# CREDIT RISK / DELINQUENCY MODEL (PRODUCTION VERSION)
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE

# =========================
# 1. PATH SETUP (IMPORTANT FOR GITHUB)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Delinquency_prediction_dataset.xlsx")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_excel(DATA_PATH)

print("\n===== DATA LOADED =====")
print(df.shape)
print(df.head())

# =========================
# 3. MISSING VALUE HANDLING
# =========================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\n===== MISSING VALUES FIXED =====")

# =========================
# 4. ENCODING
# =========================

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\n===== ENCODING DONE =====")

# =========================
# 5. SPLIT FEATURES & TARGET
# =========================

target_col = "Delinquent_Account"

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 6. SMOTE (HANDLE IMBALANCE)
# =========================

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n===== AFTER SMOTE =====")
print(y_train_resampled.value_counts())

# =========================
# 7. MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)

print("\n===== MODEL TRAINED =====")

# =========================
# 8. PREDICTIONS
# =========================

y_pred = model.predict(X_test)

print("\n===== SAMPLE PREDICTIONS =====")
print(y_pred[:10])

# =========================
# 9. EVALUATION
# =========================

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(VISUAL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

# =========================
# 10. FEATURE IMPORTANCE
# =========================

importances = model.feature_importances_

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")

fi_path = os.path.join(VISUAL_DIR, "feature_importance.png")
plt.savefig(fi_path)
plt.show()

# =========================
# 11. SAVE OUTPUT (FIXED - NO DUPLICATES)
# =========================

output = X_test.copy()
output["Actual"] = y_test.values
output["Predicted"] = y_pred

output_file = os.path.join(OUTPUT_DIR, "delinquency_predictions_output.xlsx")

# overwrite cleanly
if os.path.exists(output_file):
    os.remove(output_file)

output.to_excel(output_file, index=False)

print("\n===== OUTPUT SAVED =====")
print("Saved at:", output_file)