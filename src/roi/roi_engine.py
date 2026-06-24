"""Phase 8 – ROI Engine"""
import sys, logging
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import ANNUAL_GROWTH_RATES, RENTAL_YIELD_RATES, FORECAST_YEARS
log = logging.getLogger(__name__)

class ROIEngine:
    def calculate(self, current_price, city="default", forecast_years=5,
                  custom_growth_rate=None, custom_rental_yield=None):
        g = custom_growth_rate or ANNUAL_GROWTH_RATES.get(city, ANNUAL_GROWTH_RATES["default"])
        r = custom_rental_yield or RENTAL_YIELD_RATES.get(city, RENTAL_YIELD_RATES["default"])
        fv5 = current_price * ((1 + g) ** forecast_years)
        cagr = (((fv5 / current_price) ** (1 / forecast_years)) - 1) * 100
        annual_rent = current_price * r
        primary_roi = round(cagr + r * 100, 2)
        projections = {}
        for yy in FORECAST_YEARS:
            fv = current_price * ((1 + g) ** yy)
            projections[f"{yy}_year"] = {
                "future_price": round(fv, -3),
                "capital_gain": round(fv - current_price, -3),
                "cumulative_rental": round(annual_rent * yy, -3),
                "total_return_pct": round((fv - current_price + annual_rent * yy) / current_price * 100, 2),
            }
        return {
            "current_price": current_price, "city": city,
            "annual_growth_rate_pct": round(g * 100, 2),
            "rental_yield_pct": round(r * 100, 2),
            "monthly_rental_estimate": round(annual_rent / 12, 2),
            "future_value_5yr": round(fv5, -3),
            "total_roi_5yr_pct": round((fv5 - current_price) / current_price * 100, 2),
            "annualised_roi_pct": round(cagr, 2),
            "primary_roi_pct": primary_roi,
            "projections": projections,
        }
