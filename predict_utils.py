"""Shared loading + prediction code used by BOTH the Gradio and Flask apps."""
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "housing_logistic_model.pkl"
INFO_PATH = BASE_DIR / "models" / "model_info.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run `python train_model.py` first to create models/."
    )

model = joblib.load(MODEL_PATH)
info = json.loads(INFO_PATH.read_text())

NUMERIC = info["numeric_features"]
OCEAN_OPTIONS = info["ocean_proximity_options"]
DEFAULTS = info["defaults"]
THRESHOLD = info["price_threshold"]
ACCURACY = info["metrics"]["accuracy"]


def predict(values: dict) -> dict:
    """values: dict with the 8 numeric features + ocean_proximity."""
    row = {c: float(values[c]) for c in NUMERIC}
    row["ocean_proximity"] = values["ocean_proximity"]
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[0]  # [P(Low), P(High)]
    label = "High" if proba[1] >= 0.5 else "Low"
    return {
        "label": label,
        "p_low": float(proba[0]),
        "p_high": float(proba[1]),
        "threshold": THRESHOLD,
    }
