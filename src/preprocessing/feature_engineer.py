"""
Phase 5 – Feature Engineering Pipeline
- Encoding (Label + One-Hot)
- Scaling (Standard + MinMax)
- Feature Selection (correlation + tree-based)
- Saves encoders, scaler, and feature list
"""

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    PROCESSED_DATA_PATH, FEATURES_DATA_PATH,
    SCALER_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN,
    RANDOM_STATE, MODELS_DIR
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class FeatureEngineer:
    """Full feature engineering pipeline."""

    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.feature_columns: list = []

    # ── Encoding ─────────────────────────────────────────────────────────────
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Encoding categorical features …")
        df = df.copy()

        # Low-cardinality → One-Hot
        ohe_cols = ["property_type", "furnished_status", "parking_availability"]
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)

        # High-cardinality → Label Encode
        le_cols = [c for c in ["city", "area", "locality"]
                   if c in df.columns]
        for col in le_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        return df

    # ── Scaling ───────────────────────────────────────────────────────────────
    def scale_numericals(self, df: pd.DataFrame,
                          fit: bool = True) -> pd.DataFrame:
        log.info("Scaling numerical features …")
        df = df.copy()
        num_present = [c for c in NUMERICAL_FEATURES if c in df.columns]
        if fit:
            df[num_present] = self.scaler.fit_transform(df[num_present])
        else:
            df[num_present] = self.scaler.transform(df[num_present])
        return df

    # ── Feature Selection ─────────────────────────────────────────────────────
    def select_features(self, X: pd.DataFrame,
                         y: pd.Series) -> pd.DataFrame:
        log.info("Running tree-based feature selection …")
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        # Keep features with importance > mean importance
        threshold = importances.mean()
        selected = importances[importances >= threshold].index.tolist()
        log.info("Selected %d / %d features", len(selected), len(X.columns))
        self.feature_columns = selected
        return X[selected]

    # ── Full Pipeline ─────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        df = self.encode_categoricals(df)

        # Drop non-feature columns
        drop_cols = [c for c in ["property_id", "pincode", "price_per_sqft"]
                     if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        y = df.pop(TARGET_COLUMN)
        df = self.scale_numericals(df, fit=True)

        X = self.select_features(df, y)
        log.info("Final feature matrix shape: %s", X.shape)
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.encode_categoricals(df)
        drop_cols = [c for c in ["property_id", "pincode", "price_per_sqft"]
                     if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])
        df = self.scale_numericals(df, fit=False)
        # Keep only trained features, fill missing with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_columns]

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(FEATURE_COLUMNS_PATH, "wb") as f:
            pickle.dump(self.feature_columns, f)
        log.info("Encoders, scaler, and feature columns saved.")

    @classmethod
    def load(cls) -> "FeatureEngineer":
        fe = cls()
        with open(SCALER_PATH, "rb") as f:
            fe.scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            fe.label_encoders = pickle.load(f)
        with open(FEATURE_COLUMNS_PATH, "rb") as f:
            fe.feature_columns = pickle.load(f)
        log.info("FeatureEngineer loaded from disk.")
        return fe


def run_feature_engineering(
    input_path: Path = PROCESSED_DATA_PATH,
    output_path: Path = FEATURES_DATA_PATH,
) -> tuple:
    log.info("Loading processed data …")
    df = pd.read_csv(input_path)

    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    fe.save()

    # Save feature matrix
    combined = X.copy()
    combined[TARGET_COLUMN] = y.values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    log.info("Feature data saved → %s", output_path)

    return X, y, fe


if __name__ == "__main__":
    X, y, fe = run_feature_engineering()
    print("Feature columns:", fe.feature_columns[:10])
    print("X shape:", X.shape)
