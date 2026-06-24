"""
Phase 10 – Buy / Hold / Sell Recommendation Engine
Rules engine combining ROI, Risk Score, and Valuation Status.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RECOMMENDATION_RULES

log = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Decision matrix:

    BUY  → primary_roi ≥ 15%  AND risk_score ≤ 40  (OR undervalued)
    HOLD → primary_roi ≥  8%  AND risk_score ≤ 65
    SELL → otherwise
    """

    def recommend(
        self,
        primary_roi_pct: float,
        risk_score: float,
        valuation_status: str,
        risk_category: str,
    ) -> dict:

        # BUY logic
        buy_roi    = primary_roi_pct >= RECOMMENDATION_RULES["BUY"]["roi_min"]
        buy_risk   = risk_score      <= RECOMMENDATION_RULES["BUY"]["risk_max"]
        is_under   = valuation_status == "undervalued"

        # SELL logic
        sell_roi   = primary_roi_pct < RECOMMENDATION_RULES["SELL"]["roi_max"]
        sell_risk  = risk_score      >= RECOMMENDATION_RULES["SELL"]["risk_min"]

        if (buy_roi and buy_risk) or (is_under and buy_risk):
            action = "BUY"
            color  = "#22c55e"
            emoji  = "✅"
            confidence = self._confidence("BUY", primary_roi_pct, risk_score)
            reason = self._reason("BUY", primary_roi_pct, risk_score,
                                  valuation_status)
        elif sell_roi or sell_risk:
            action = "SELL"
            color  = "#ef4444"
            emoji  = "🔴"
            confidence = self._confidence("SELL", primary_roi_pct, risk_score)
            reason = self._reason("SELL", primary_roi_pct, risk_score,
                                  valuation_status)
        else:
            action = "HOLD"
            color  = "#f59e0b"
            emoji  = "⏸️"
            confidence = self._confidence("HOLD", primary_roi_pct, risk_score)
            reason = self._reason("HOLD", primary_roi_pct, risk_score,
                                  valuation_status)

        return {
            "recommendation": action,
            "color": color,
            "emoji": emoji,
            "confidence_pct": confidence,
            "reason": reason,
            "inputs": {
                "roi_pct": primary_roi_pct,
                "risk_score": risk_score,
                "valuation": valuation_status,
                "risk_category": risk_category,
            }
        }

    def _confidence(self, action: str, roi: float,
                    risk: float) -> float:
        """Rough confidence based on how far inputs are from thresholds."""
        if action == "BUY":
            roi_conf  = min((roi - 15) / 15, 1.0) if roi > 15 else 0
            risk_conf = min((40 - risk) / 40, 1.0) if risk < 40 else 0
            return round((0.5 * roi_conf + 0.5 * risk_conf) * 100, 1)
        elif action == "SELL":
            roi_conf  = min((8 - roi) / 8, 1.0)  if roi < 8  else 0
            risk_conf = min((risk - 65) / 35, 1.0) if risk > 65 else 0
            return round((0.5 * roi_conf + 0.5 * risk_conf) * 100, 1)
        else:
            # HOLD: middle band confidence
            return round(60.0, 1)

    def _reason(self, action: str, roi: float,
                risk: float, valuation: str) -> str:
        base = {
            "BUY": (
                f"Property shows strong investment potential with {roi:.1f}% "
                f"annualised ROI and a low risk score of {risk:.0f}/100. "
                f"Valuation: {valuation}."
            ),
            "HOLD": (
                f"Moderate return of {roi:.1f}% ROI with a risk score "
                f"of {risk:.0f}/100. Monitor for 6–12 months before deciding."
            ),
            "SELL": (
                f"Low ROI of {roi:.1f}% or high risk score of {risk:.0f}/100 "
                f"suggests unfavourable investment profile. Consider exit."
            ),
        }
        return base[action]
