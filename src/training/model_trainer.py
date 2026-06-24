"""
Phase 6 – Model Training Pipeline
Trains and evaluates: LinearRegression, RandomForest, GradientBoosting,
XGBoost, CatBoost, LightGBM
Performs: Cross-Validation, Hyperparameter Tuning (RandomizedSearch)
Saves best model by R² score.
"""

import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    FEATURES_DATA_PATH, BEST_MODEL_PATH, MODELS_DIR,
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, CV_FOLDS
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Model Registry ─────────────────────────────────────────────────────────────
def get_model_candidates() -> Dict[str, Any]:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, learning_rate=0.08,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150, learning_rate=0.08,
            num_leaves=31, random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=150, learning_rate=0.08,
            depth=6, random_seed=RANDOM_STATE,
            verbose=0
        ),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


# ── Hyperparameter tuning for best candidate ─────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.05, 0.08, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}


def tune_best_model(X_train, y_train) -> Any:
    log.info("Hyperparameter tuning XGBoost …")
    base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=0)
    rs = RandomizedSearchCV(
        base, XGB_PARAM_GRID,
        n_iter=5, cv=3,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    rs.fit(X_train, y_train)
    log.info("Best XGB params: %s | CV R²: %.4f",
             rs.best_params_, rs.best_score_)
    return rs.best_estimator_


# ── Training Pipeline ─────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = ""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, path: Path = FEATURES_DATA_PATH):
        df = pd.read_csv(path)
        y = df.pop(TARGET_COLUMN)
        X = df
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=TEST_SIZE,
                             random_state=RANDOM_STATE)
        log.info("Train: %d | Test: %d", len(self.X_train), len(self.X_test))
        return self

    def train_all(self):
        candidates = get_model_candidates()
        for name, model in candidates.items():
            log.info("Training %s …", name)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            metrics = compute_metrics(self.y_test.values, y_pred)

            # Cross validation (sequential execution to avoid Windows pickle issues)
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=CV_FOLDS, scoring="r2", n_jobs=1
            )
            metrics["CV_R2_mean"] = round(cv_scores.mean(), 4)
            metrics["CV_R2_std"]  = round(cv_scores.std(), 4)

            self.results[name] = {
                "model": model,
                "metrics": metrics,
            }
            log.info("%s → R²=%.4f | RMSE=%.0f | MAE=%.0f",
                     name, metrics["R2"], metrics["RMSE"], metrics["MAE"])

        return self

    def select_best(self):
        best_name = max(
            self.results,
            key=lambda k: self.results[k]["metrics"]["R2"]
        )
        log.info("🏆 Best model: %s (R²=%.4f)",
                 best_name, self.results[best_name]["metrics"]["R2"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]

        # Fine-tune if best is XGBoost or close to it
        if "XGBoost" in best_name or self.results[best_name]["metrics"]["R2"] >= 0.80:
            tuned = tune_best_model(self.X_train, self.y_train)
            y_pred = tuned.predict(self.X_test)
            tuned_metrics = compute_metrics(self.y_test.values, y_pred)
            if tuned_metrics["R2"] > self.results[best_name]["metrics"]["R2"]:
                self.best_model = tuned
                self.results["XGBoost (Tuned)"] = {
                    "model": tuned,
                    "metrics": tuned_metrics
                }
                self.best_model_name = "XGBoost (Tuned)"
                log.info("Tuned model is better. R²=%.4f", tuned_metrics["R2"])
        return self

    def save_best(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(self.best_model, f)
        log.info("Best model saved → %s", BEST_MODEL_PATH)

        # Save metrics report
        report = {
            name: data["metrics"]
            for name, data in self.results.items()
        }
        report_path = MODELS_DIR / "model_comparison.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Model comparison saved → %s", report_path)
        return self

    def get_comparison_df(self) -> pd.DataFrame:
        rows = []
        for name, data in self.results.items():
            row = {"Model": name}
            row.update(data["metrics"])
            rows.append(row)
        return pd.DataFrame(rows).sort_values("R2", ascending=False)


def load_best_model():
    with open(BEST_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def run_training_pipeline(data_path: Path = FEATURES_DATA_PATH) -> ModelTrainer:
    trainer = (
        ModelTrainer()
        .load_data(data_path)
        .train_all()
        .select_best()
        .save_best()
    )
    print("\n=== Model Comparison ===")
    print(trainer.get_comparison_df().to_string(index=False))
    return trainer


if __name__ == "__main__":
    run_training_pipeline()
