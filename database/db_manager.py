"""
SQLite Database Layer
Modular design – swap with PostgreSQL by changing DATABASE_URL.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, MetaData, Table, inspect
)
from sqlalchemy.sql import func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DATABASE_URL, DATABASE_DIR

log = logging.getLogger(__name__)


class DatabaseManager:
    """SQLAlchemy-backed database manager."""

    def __init__(self, url: str = DATABASE_URL):
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, echo=False)
        self.metadata = MetaData()
        self._define_tables()
        self.metadata.create_all(self.engine)
        log.info("Database initialised at %s", url)

    def _define_tables(self):
        self.predictions_table = Table(
            "predictions", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime, server_default=func.now()),
            Column("city", String(100)),
            Column("area_sqft", Float),
            Column("bhk", Integer),
            Column("property_age", Integer),
            Column("predicted_price", Float),
            Column("valuation_status", String(50)),
            Column("risk_score", Float),
            Column("risk_category", String(50)),
            Column("recommendation", String(20)),
            Column("roi_pct", Float),
            extend_existing=True,
        )

        self.market_insights_table = Table(
            "market_insights", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime, server_default=func.now()),
            Column("city", String(100)),
            Column("avg_price", Float),
            Column("median_price", Float),
            Column("avg_roi", Float),
            Column("avg_risk", Float),
            Column("top_locality", String(100)),
            extend_existing=True,
        )

    def save_prediction(self, data: dict) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                self.predictions_table.insert().values(**data)
            )
            conn.commit()
            return result.lastrowid

    def get_recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(
                self.predictions_table.select()
                .order_by(self.predictions_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = result.fetchall()
            if rows:
                return pd.DataFrame(rows, columns=result.keys())
            return pd.DataFrame()

    def get_prediction_count(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                self.predictions_table.count()
            )
            return result.scalar() or 0
