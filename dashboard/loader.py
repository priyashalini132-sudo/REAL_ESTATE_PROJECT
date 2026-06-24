"""
dashboard/loader.py – Cached model/data loading for Streamlit
"""
import sys, pickle, logging
from pathlib import Path
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from config.settings import (
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH,
    PROCESSED_DATA_PATH, FEATURES_DATA_PATH
)
log = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

@st.cache_data(show_spinner="Loading features…")
def load_features():
    df = pd.read_csv(FEATURES_DATA_PATH)
    return df

def models_ready() -> bool:
    return all(p.exists() for p in [BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH])

def data_ready() -> bool:
    return PROCESSED_DATA_PATH.exists()
