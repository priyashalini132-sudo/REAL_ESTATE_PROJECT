"""
Phase 13 – Property Ranking Engine
Ranks properties by composite investment attractiveness score.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Scoring weights
RANK_WEIGHTS = {
    "roi":        0.35,
    "risk":       0.25,   # inverted
    "valuation":  0.20,
    "growth":     0.20,
}

VALUATION_SCORES = {
    "undervalued": 100,
    "fair_value":  60,
    "overvalued":  20,
    "unknown":     40,
}


class RankingEngine:
    """Scores and ranks properties by investment attractiveness."""

    def score_property(
        self,
        roi_pct: float,
        risk_score: float,
        valuation_status: str,
        infra_growth_score: float,
    ) -> float:
        """
        Returns composite score 0-100.
        """
        # Normalise ROI to 0-100 (cap at 30% annual)
        roi_norm = float(np.clip(roi_pct / 30 * 100, 0, 100))

        # Invert risk (lower risk = better)
        risk_norm = float(np.clip(100 - risk_score, 0, 100))

        # Valuation bonus
        val_norm = float(VALUATION_SCORES.get(valuation_status, 40))

        # Growth
        growth_norm = float(np.clip(infra_growth_score, 0, 100))

        score = (
            roi_norm    * RANK_WEIGHTS["roi"] +
            risk_norm   * RANK_WEIGHTS["risk"] +
            val_norm    * RANK_WEIGHTS["valuation"] +
            growth_norm * RANK_WEIGHTS["growth"]
        )
        return round(float(score), 2)

    def rank_dataframe(self, df: pd.DataFrame,
                        roi_col: str = "primary_roi_pct",
                        risk_col: str = "risk_score",
                        val_col: str = "valuation_status",
                        infra_col: str = "infrastructure_growth_score",
                        top_n: int = 20) -> pd.DataFrame:
        """Rank a dataframe of properties."""
        df = df.copy()
        df["investment_score"] = df.apply(
            lambda r: self.score_property(
                r[roi_col], r[risk_col], r[val_col], r[infra_col]
            ), axis=1
        )
        df["rank"] = df["investment_score"].rank(
            ascending=False, method="min"
        ).astype(int)
        return df.sort_values("investment_score", ascending=False).head(top_n)

    def grade(self, score: float) -> tuple:
        """Letter grade + color."""
        if score >= 80:
            return "A+", "#22c55e"
        elif score >= 70:
            return "A",  "#4ade80"
        elif score >= 60:
            return "B+", "#86efac"
        elif score >= 50:
            return "B",  "#f59e0b"
        elif score >= 40:
            return "C",  "#f97316"
        else:
            return "D",  "#ef4444"
