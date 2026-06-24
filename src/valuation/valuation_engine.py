"""
Phase 7 – Property Valuation Engine
Determines Fair Market Value, Overvalued, and Undervalued properties.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import VALUATION_TOLERANCE

log = logging.getLogger(__name__)


class ValuationEngine:
    """
    Compares actual/listed price against the model's predicted fair value.
    """

    def __init__(self, tolerance: float = VALUATION_TOLERANCE):
        self.tolerance = tolerance  # e.g. 0.15 = ±15%

    def classify(self, actual_price: float,
                 predicted_price: float) -> dict:
        """
        Returns valuation status + percentage deviation.

        Valuation:
            undervalued  → actual < predicted * (1 - tol)
            overvalued   → actual > predicted * (1 + tol)
            fair_value   → within tolerance band
        """
        if predicted_price <= 0:
            return {"status": "unknown", "deviation_pct": 0.0,
                    "fair_value": predicted_price}

        deviation = (actual_price - predicted_price) / predicted_price
        deviation_pct = round(deviation * 100, 2)

        lower = predicted_price * (1 - self.tolerance)
        upper = predicted_price * (1 + self.tolerance)

        if actual_price < lower:
            status = "undervalued"
        elif actual_price > upper:
            status = "overvalued"
        else:
            status = "fair_value"

        return {
            "status": status,
            "deviation_pct": deviation_pct,
            "fair_value": round(predicted_price, -3),
            "lower_bound": round(lower, -3),
            "upper_bound": round(upper, -3),
            "potential_gain": round(predicted_price - actual_price, -3),
        }

    def classify_batch(self, df: pd.DataFrame,
                       actual_col: str = "price",
                       predicted_col: str = "predicted_price") -> pd.DataFrame:
        results = df.apply(
            lambda r: self.classify(r[actual_col], r[predicted_col]),
            axis=1
        )
        return pd.concat([df, pd.DataFrame(results.tolist())], axis=1)


def get_valuation_label_color(status: str) -> tuple:
    """Returns (label, emoji, color) for UI display."""
    mapping = {
        "undervalued": ("Undervalued 🟢", "🟢", "#22c55e"),
        "overvalued":  ("Overvalued 🔴", "🔴", "#ef4444"),
        "fair_value":  ("Fair Value 🟡", "🟡", "#f59e0b"),
        "unknown":     ("Unknown ⚪", "⚪", "#6b7280"),
    }
    return mapping.get(status, mapping["unknown"])
