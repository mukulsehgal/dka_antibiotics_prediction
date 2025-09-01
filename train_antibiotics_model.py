#!/usr/bin/env python3
"""
Train a prediction model for 'Antibiotics Given' using patient visit data.
Handles mixed-type categorical + numeric columns.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------- CONFIG ----------
INPUT_FILE = "merged patients final v1.xlsx"
OUTPUT_MODEL = "antibiotics_model.pkl"
TARGET_COL = "ANTIBIOTICS GIVEN"
EXCLUDE_COLS = ["VISIT #", "PATIENT #", "LENGTH OF STAY", "CCI"]

# ---------- Load Data ----------
print("[INFO] Loading dataset...")
df = pd.read_excel(INPUT_FILE)

# Normalize column names
df.columns = df.columns.astype(str).str.strip().str.upper()

# Ensure target column exists
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

# Drop excluded columns
X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors="ignore")

# Target variable (encode Yes/No into 1/0)
y = X[TARGET_COL].map({"Yes": 1, "No": 0})
X = X.drop(columns=[TARGET_COL])

# Fill missing
X = X.fillna("MISSING")

# Split categorical vs numeric
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# --- Force categorical columns to string type ---
for col in categorical_cols:
    X[col] = X[col].astype(str)

print(f"[INFO] Found {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns.")

# ---------- Preprocessor ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ---------- Pipeline ----------
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[INFO] Training model...")
pipeline.fit(X_train, y_train)

# ---------- Evaluation ----------
print("[INFO] Evaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Save Model ----------
joblib.dump(pipeline, OUTPUT_MODEL)
print(f"[INFO] Model saved to {OUTPUT_MODEL}")
