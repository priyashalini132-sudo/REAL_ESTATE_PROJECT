"""
Global Configuration Settings
AI-Powered Real Estate Intelligence Platform
"""

import os
from pathlib import Path

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"

# ── Data Paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH = DATA_DIR / "raw" / "real_estate_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "real_estate_processed.csv"
FEATURES_DATA_PATH = DATA_DIR / "processed" / "real_estate_features.csv"

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/real_estate.db"

# ── Model Settings ────────────────────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1

# ── Feature Columns ───────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "property_type", "city", "area", "locality",
    "furnished_status", "parking_availability"
]

NUMERICAL_FEATURES = [
    "bhk", "bathrooms", "balcony", "floor_number", "total_floors",
    "area_sqft", "property_age",
    "hospital_distance_km", "school_distance_km",
    "metro_distance_km", "mall_distance_km", "airport_distance_km",
    "population_density", "crime_index", "infrastructure_growth_score"
]

TARGET_COLUMN = "price"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ── Risk Engine ───────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low": 35,
    "medium": 65,
    "high": 100,
}

RISK_WEIGHTS = {
    "crime_index": 0.30,
    "property_age": 0.20,
    "market_volatility": 0.20,
    "infrastructure_growth_score": 0.15,
    "demand_trend": 0.15,
}

# ── ROI Engine ────────────────────────────────────────────────────────────────
ANNUAL_GROWTH_RATES = {
    "Mumbai": 0.085,
    "Delhi": 0.078,
    "Bangalore": 0.092,
    "Hyderabad": 0.088,
    "Chennai": 0.075,
    "Pune": 0.082,
    "Kolkata": 0.065,
    "Ahmedabad": 0.072,
    "Jaipur": 0.068,
    "Noida": 0.080,
    "default": 0.070,
}

RENTAL_YIELD_RATES = {
    "Mumbai": 0.030,
    "Delhi": 0.035,
    "Bangalore": 0.040,
    "Hyderabad": 0.038,
    "Chennai": 0.032,
    "Pune": 0.036,
    "default": 0.033,
}

# ── Recommendation Thresholds ─────────────────────────────────────────────────
RECOMMENDATION_RULES = {
    "BUY": {"roi_min": 15.0, "risk_max": 40, "valuation": "undervalued"},
    "HOLD": {"roi_min": 8.0, "risk_max": 65},
    "SELL": {"roi_max": 8.0, "risk_min": 65},
}

# ── Valuation Engine ──────────────────────────────────────────────────────────
VALUATION_TOLERANCE = 0.15  # ±15% from fair market value

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_YEARS = [1, 3, 5]

# ── Dashboard ─────────────────────────────────────────────────────────────────
APP_TITLE = "🏠 AI Real Estate Intelligence Platform"
APP_SUBTITLE = "Production-Grade Property Valuation & Investment Analysis"
APP_ICON = "🏠"
THEME_COLOR = "#6C63FF"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = BASE_DIR / "reports" / "platform.log"
