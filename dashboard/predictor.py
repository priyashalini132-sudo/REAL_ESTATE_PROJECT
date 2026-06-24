"""
Prediction helper — builds a feature row and runs the model.
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from config.settings import (
    SCALER_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)


def predict_price(inputs: dict, model, scaler, encoders, feature_cols) -> float:
    """
    inputs: dict with all property features (raw, before encoding).
    Returns predicted price in INR.
    """
    df = pd.DataFrame([inputs])

    # One-hot encode low-cardinality cats
    ohe_cols = ["property_type", "furnished_status", "parking_availability"]
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns],
                        drop_first=False, dtype=int)

    # Label encode high-cardinality
    le_cols = ["city", "area", "locality"]
    for col in le_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # Drop non-feature cols
    drop = [c for c in ["property_id", "pincode", "price_per_sqft", "price"]
            if c in df.columns]
    df.drop(columns=drop, inplace=True, errors="ignore")

    # Scale numericals
    num_present = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_present] = scaler.transform(df[num_present])

    # Align to trained feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    pred = model.predict(df)[0]
    return float(pred)
