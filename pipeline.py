"""
Master Pipeline Runner
Executes all phases in sequence to prepare the platform for the dashboard.
Run once before launching the Streamlit app.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("PIPELINE")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURES_DATA_PATH


def run():
    log.info("=" * 60)
    log.info("   AI REAL ESTATE INTELLIGENCE PLATFORM — SETUP PIPELINE")
    log.info("=" * 60)

    # Phase 2: Generate dataset
    log.info("[Phase 2] Data generation …")
    from src.preprocessing.data_generator import generate_dataset, save_dataset
    df_raw = generate_dataset(n=20_000)
    save_dataset(df_raw, RAW_DATA_PATH)
    log.info("  ✔ Raw dataset: %d rows", len(df_raw))

    # Phase 3: Clean data
    log.info("[Phase 3] Data cleaning …")
    from src.preprocessing.data_cleaner import run_cleaning_pipeline
    df_clean = run_cleaning_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    log.info("  ✔ Cleaned dataset: %d rows", len(df_clean))

    # Phase 5: Feature engineering
    log.info("[Phase 5] Feature engineering …")
    from src.preprocessing.feature_engineer import run_feature_engineering
    X, y, fe = run_feature_engineering(PROCESSED_DATA_PATH, FEATURES_DATA_PATH)
    log.info("  ✔ Features: %d columns", X.shape[1])

    # Phase 6: Model training
    log.info("[Phase 6] Model training …")
    from src.training.model_trainer import run_training_pipeline
    trainer = run_training_pipeline(FEATURES_DATA_PATH)
    log.info("  ✔ Best model: %s", trainer.best_model_name)

    # Initialise DB
    log.info("[DB] Initialising database …")
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    log.info("  ✔ Database ready")

    log.info("=" * 60)
    log.info("   PIPELINE COMPLETE — Launch: streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
