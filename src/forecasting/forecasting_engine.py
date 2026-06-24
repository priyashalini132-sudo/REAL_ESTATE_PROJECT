"""
Phase 11 – Future Price Forecasting Engine
Uses city-level historical growth rates + CAGR projections.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import ANNUAL_GROWTH_RATES, FORECAST_YEARS

log = logging.getLogger(__name__)


class ForecastingEngine:
    """Multi-year property price forecaster."""

    def forecast(
        self,
        current_price: float,
        city: str = "default",
        custom_growth_rate: float = None,
    ) -> dict:
        """
        Returns year-by-year price forecast with confidence intervals.
        """
        growth_rate = custom_growth_rate or ANNUAL_GROWTH_RATES.get(
            city, ANNUAL_GROWTH_RATES["default"]
        )

        # Simulate volatility as ±2% standard deviation per year
        years = list(range(0, max(FORECAST_YEARS) + 1))
        prices_base = []
        prices_upper = []
        prices_lower = []

        for y in years:
            base = current_price * ((1 + growth_rate) ** y)
            vol = (0.02 * y)                    # growing uncertainty
            prices_base.append(round(base, -3))
            prices_upper.append(round(base * (1 + vol), -3))
            prices_lower.append(round(base * (1 - vol), -3))

        # Key milestones
        milestones = {}
        for yy in FORECAST_YEARS:
            milestones[f"{yy}_year"] = {
                "price": prices_base[yy],
                "price_upper": prices_upper[yy],
                "price_lower": prices_lower[yy],
                "total_gain": round(prices_base[yy] - current_price, -3),
                "total_roi_pct": round(
                    (prices_base[yy] - current_price) / current_price * 100, 2
                ),
            }

        return {
            "current_price": current_price,
            "city": city,
            "growth_rate_pct": round(growth_rate * 100, 2),
            "years": years,
            "prices_base": prices_base,
            "prices_upper": prices_upper,
            "prices_lower": prices_lower,
            "milestones": milestones,
        }

    def plot_forecast(self, forecast_data: dict,
                      property_name: str = "Property") -> go.Figure:
        years = forecast_data["years"]
        base   = forecast_data["prices_base"]
        upper  = forecast_data["prices_upper"]
        lower  = forecast_data["prices_lower"]

        fig = go.Figure()

        # Confidence band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(108,99,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
            hoverinfo="skip",
        ))

        # Base forecast
        fig.add_trace(go.Scatter(
            x=years, y=base,
            mode="lines+markers",
            line=dict(color="#6C63FF", width=3),
            marker=dict(size=8, color="#6C63FF"),
            name="Projected Price",
        ))

        # Milestone markers
        for yy in FORECAST_YEARS:
            if yy <= max(years):
                fig.add_annotation(
                    x=yy,
                    y=base[yy],
                    text=f"₹{base[yy]/1e5:.1f}L",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=11, color="#fff"),
                    bgcolor="#6C63FF",
                    bordercolor="#6C63FF",
                )

        fig.update_layout(
            title=f"🔮 Future Price Forecast — {property_name}",
            xaxis_title="Years from Now",
            yaxis_title="Projected Price (₹)",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        return fig
