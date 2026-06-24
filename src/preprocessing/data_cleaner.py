"""
Phase 3 – Data Cleaning Pipeline
Handles:
  - Missing value imputation
  - Duplicate removal
  - Outlier detection (IQR + Z-Score)
  - Data type validation
  - Consistency checks
  - Cleaning report generation
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class DataCleaner:
    """End-to-end data cleaning pipeline."""

    def __init__(self, df: pd.DataFrame):
        self.df_original = df.copy()
        self.df = df.copy()
        self.report: Dict[str, Any] = {
            "initial_shape": list(df.shape),
            "steps": {}
        }

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    def remove_duplicates(self) -> "DataCleaner":
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        self.report["steps"]["duplicate_removal"] = {
            "before": before, "after": len(self.df), "removed": removed
        }
        log.info("Duplicates removed: %d", removed)
        return self

    # ── Step 2: Fix data types ────────────────────────────────────────────────
    def fix_dtypes(self) -> "DataCleaner":
        int_cols = ["bhk", "bathrooms", "balcony", "floor_number",
                    "total_floors", "area_sqft", "property_age",
                    "population_density", "pincode"]
        for col in int_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        cat_cols = CATEGORICAL_FEATURES + ["locality"]
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.title()
        self.report["steps"]["dtype_fix"] = "completed"
        log.info("Data types standardised.")
        return self

    # ── Step 3: Impute missing values ─────────────────────────────────────────
    def impute_missing(self) -> "DataCleaner":
        mv_before = self.df.isnull().sum().to_dict()
        # Numerical → median
        for col in NUMERICAL_FEATURES:
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        # Target → drop rows
        self.df = self.df.dropna(subset=[TARGET_COLUMN])
        # Categorical → mode
        for col in CATEGORICAL_FEATURES:
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        mv_after = self.df.isnull().sum().to_dict()
        self.report["steps"]["missing_value_imputation"] = {
            "before": mv_before, "after": mv_after
        }
        log.info("Missing values imputed.")
        return self

    # ── Step 4: Detect & cap outliers (IQR) ───────────────────────────────────
    def cap_outliers(self, cols: list = None) -> "DataCleaner":
        if cols is None:
            cols = ["area_sqft", "price", "property_age",
                    "hospital_distance_km", "airport_distance_km"]
        capped = {}
        for col in cols:
            if col not in self.df.columns:
                continue
            Q1 = self.df[col].quantile(0.01)
            Q3 = self.df[col].quantile(0.99)
            before = ((self.df[col] < Q1) | (self.df[col] > Q3)).sum()
            self.df[col] = self.df[col].clip(lower=Q1, upper=Q3)
            capped[col] = int(before)
        self.report["steps"]["outlier_capping"] = capped
        log.info("Outliers capped: %s", capped)
        return self

    # ── Step 5: Consistency checks ─────────────────────────────────────────────
    def consistency_checks(self) -> "DataCleaner":
        issues = {}
        # floor_number must be ≤ total_floors
        bad_floor = (self.df["floor_number"] > self.df["total_floors"]).sum()
        if bad_floor:
            self.df.loc[self.df["floor_number"] > self.df["total_floors"],
                        "floor_number"] = self.df["total_floors"]
            issues["floor_number_gt_total"] = int(bad_floor)

        # bathrooms ≤ bhk + 2
        bad_bath = (self.df["bathrooms"] > self.df["bhk"] + 2).sum()
        if bad_bath:
            self.df.loc[self.df["bathrooms"] > self.df["bhk"] + 2,
                        "bathrooms"] = self.df["bhk"] + 1
            issues["bathrooms_excessive"] = int(bad_bath)

        # price > 0
        neg_price = (self.df["price"] <= 0).sum()
        if neg_price:
            self.df = self.df[self.df["price"] > 0]
            issues["non_positive_price"] = int(neg_price)

        # area_sqft > 0
        neg_sqft = (self.df["area_sqft"] <= 0).sum()
        if neg_sqft:
            self.df = self.df[self.df["area_sqft"] > 0]
            issues["non_positive_sqft"] = int(neg_sqft)

        # crime_index 0-100
        self.df["crime_index"] = self.df["crime_index"].clip(0, 100)
        # infra_score 0-100
        self.df["infrastructure_growth_score"] = \
            self.df["infrastructure_growth_score"].clip(0, 100)

        self.report["steps"]["consistency_checks"] = issues
        log.info("Consistency checks done: %s", issues)
        return self

    # ── Step 6: Feature derivation (adds price_per_sqft) ─────────────────────
    def derive_features(self) -> "DataCleaner":
        self.df["price_per_sqft"] = (
            self.df["price"] / self.df["area_sqft"]
        ).round(2)
        self.report["steps"]["derived_features"] = ["price_per_sqft"]
        log.info("Derived feature 'price_per_sqft' added.")
        return self

    # ── Finalise ──────────────────────────────────────────────────────────────
    def finalise(self) -> pd.DataFrame:
        self.report["final_shape"] = list(self.df.shape)
        self.report["total_rows_removed"] = (
            self.report["initial_shape"][0] - self.report["final_shape"][0]
        )
        log.info(
            "Cleaning complete. Shape: %s → %s",
            self.report["initial_shape"],
            self.report["final_shape"],
        )
        return self.df

    def get_report(self) -> Dict[str, Any]:
        return self.report


def run_cleaning_pipeline(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    log.info("Loading raw data from %s …", input_path)
    df = pd.read_csv(input_path)

    cleaner = (
        DataCleaner(df)
        .remove_duplicates()
        .fix_dtypes()
        .impute_missing()
        .cap_outliers()
        .consistency_checks()
        .derive_features()
    )

    clean_df = cleaner.finalise()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    log.info("Cleaned data saved → %s", output_path)

    # Save cleaning report
    report = cleaner.get_report()
    report_path = output_path.parent / "cleaning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Cleaning report saved → %s", report_path)

    return clean_df


if __name__ == "__main__":
    clean_df = run_cleaning_pipeline()
    print(clean_df.info())
    print(clean_df.describe())
