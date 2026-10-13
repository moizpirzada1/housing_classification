"""Train a Logistic Regression classifier on housing.csv and save everything
needed for deployment into the models/ folder.

Task: predict whether a district's median house value is HIGH (above the
dataset median) or LOW. Logistic Regression needs a categorical target, and the
CSV only has a numeric price, so we convert it into two balanced classes.

Usage:
    python train_model.py
    python train_model.py "C:\\Users\\R Y Z E N\\Desktop\\housing.csv"
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data" / "housing.csv"

NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CATEGORICAL = ["ocean_proximity"]
PRICE_COL = "median_house_value"

df = pd.read_csv(CSV_PATH)
print(f"Loaded {CSV_PATH} -> {df.shape[0]} rows, {df.shape[1]} columns")

# ---- Build the classification target -------------------------------------
threshold = float(df[PRICE_COL].median())
df["price_class"] = (df[PRICE_COL] > threshold).astype(int)  # 1 = High, 0 = Low
print(f"Price threshold (median): {threshold:,.0f}")
print(df["price_class"].value_counts().rename({0: "Low", 1: "High"}))

X = df[NUMERIC + CATEGORICAL]
y = df["price_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Preprocessing + model in ONE pipeline --------------------------------
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

# ---- Evaluate -------------------------------------------------------------
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, target_names=["Low", "High"], output_dict=True)
cm = confusion_matrix(y_test, pred).tolist()
print(f"\nAccuracy: {acc:.4f}\n")
print(classification_report(y_test, pred, target_names=["Low", "High"]))
print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)

# ---- Save deployment artifacts -------------------------------------------
joblib.dump(model, MODEL_DIR / "housing_logistic_model.pkl")

meta = {
    "numeric_features": NUMERIC,
    "categorical_features": CATEGORICAL,
    "ocean_proximity_options": sorted(df["ocean_proximity"].dropna().unique().tolist()),
    "price_threshold": threshold,
    "class_names": {"0": "Low", "1": "High"},
    "defaults": {c: float(df[c].median()) for c in NUMERIC},
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
    "metrics": {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    },
}
(MODEL_DIR / "model_info.json").write_text(json.dumps(meta, indent=2))
print("\nSaved: models/housing_logistic_model.pkl and models/model_info.json")
