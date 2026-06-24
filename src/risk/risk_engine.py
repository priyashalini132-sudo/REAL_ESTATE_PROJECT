"""
Phase 9 – Risk Score Engine
Computes a composite risk score (0-100) from:
  - Crime Index
  - Property Age
  - Market Volatility (proxy: city-level std deviation)
  - Infrastructure Growth (inverted – high growth = low risk)
  - Demand Trend (derived from price_per_sqft vs city median)
"""

import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RISK_WEIGHTS, RISK_THRESHOLDS

log = logging.getLogger(__name__)

# City-level market volatility (historical proxy, 0-100)
CITY_VOLATILITY = {
    "Mumbai": 45, "Delhi": 52, "Bangalore": 38, "Hyderabad": 35,
    "Chennai": 42, "Pune": 40, "Kolkata": 60, "Ahmedabad": 48,
    "Jaipur": 55, "Noida": 50, "default": 50,
}


class RiskEngine:
    """Composite property risk scorer."""

    # ── Individual component scorers ──────────────────────────────────────────
    def _crime_score(self, crime_index: float) -> float:
        """Direct pass-through (already 0-100)."""
        return float(np.clip(crime_index, 0, 100))

    def _age_score(self, property_age: int) -> float:
        """Older property → higher risk (max at 40+ years)."""
        return float(np.clip((property_age / 40) * 100, 0, 100))

    def _volatility_score(self, city: str) -> float:
        return float(CITY_VOLATILITY.get(city, CITY_VOLATILITY["default"]))

    def _infra_score(self, infra_growth: float) -> float:
        """High infra growth → LOW risk → invert."""
        return float(np.clip(100 - infra_growth, 0, 100))

    def _demand_trend_score(self, price_per_sqft: float,
                             city_median_ppsf: float) -> float:
        """
        If current ppsf is well above city median → oversupply risk.
        If below → undervalued demand zone → lower risk.
        """
        if city_median_ppsf <= 0:
            return 50.0
        ratio = price_per_sqft / city_median_ppsf
        # ratio > 1.3 → risky area, < 0.7 → very low risk
        score = np.clip((ratio - 0.5) / 1.0 * 100, 0, 100)
        return float(score)

    # ── Composite scorer ──────────────────────────────────────────────────────
    def calculate(
        self,
        crime_index: float,
        property_age: int,
        city: str,
        infrastructure_growth_score: float,
        price_per_sqft: float = 8000,
        city_median_ppsf: float = 8000,
    ) -> dict:
        components = {
            "crime":      self._crime_score(crime_index),
            "age":        self._age_score(property_age),
            "volatility": self._volatility_score(city),
            "infra":      self._infra_score(infrastructure_growth_score),
            "demand":     self._demand_trend_score(price_per_sqft, city_median_ppsf),
        }

        weights = RISK_WEIGHTS
        composite = (
            components["crime"]      * weights["crime_index"] +
            components["age"]        * weights["property_age"] +
            components["volatility"] * weights["market_volatility"] +
            components["infra"]      * weights["infrastructure_growth_score"] +
            components["demand"]     * weights["demand_trend"]
        )
        composite = round(float(np.clip(composite, 0, 100)), 2)

        # Category
        if composite <= RISK_THRESHOLDS["low"]:
            category = "Low Risk"
            color = "#22c55e"
            emoji = "🟢"
        elif composite <= RISK_THRESHOLDS["medium"]:
            category = "Medium Risk"
            color = "#f59e0b"
            emoji = "🟡"
        else:
            category = "High Risk"
            color = "#ef4444"
            emoji = "🔴"

        return {
            "risk_score": composite,
            "risk_category": category,
            "risk_color": color,
            "risk_emoji": emoji,
            "components": components,
            "weights": weights,
        }
