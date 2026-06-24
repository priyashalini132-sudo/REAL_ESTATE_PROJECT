"""
Phase 14 – AI Real Estate Chatbot
Rule-based + ML-powered chatbot answering investment questions.
"""

import sys
import re
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)


class RealEstateChatbot:
    """
    AI chatbot for real estate investment queries.
    Uses rule-based intent detection + engine calls for quantitative answers.
    """

    INTENTS = {
        "should_invest": [
            "should i invest", "worth buying", "good investment",
            "should i buy", "invest in", "is it worth",
        ],
        "overpriced": [
            "overpriced", "overvalued", "too expensive",
            "is the price high", "is it expensive",
        ],
        "future_value": [
            "future value", "future price", "in 5 years", "in 3 years",
            "in 1 year", "appreciate", "will price increase",
        ],
        "risk": [
            "risk", "risky", "safe", "how safe", "is it safe",
            "danger", "volatile",
        ],
        "roi": [
            "roi", "return", "profit", "how much will i make",
            "rental yield", "income",
        ],
        "compare": [
            "compare", "which is better", "better property",
            "difference", "vs",
        ],
        "greeting": [
            "hello", "hi", "hey", "good morning",
            "good afternoon", "namaste",
        ],
        "help": ["help", "what can you do", "capabilities", "features"],
    }

    HELP_MSG = """
I'm your **AI Real Estate Investment Assistant** 🏠

I can help you with:
- 📊 **Investment analysis** — "Should I invest in this property?"
- 💰 **Price assessment** — "Is this property overpriced?"
- 🔮 **Future value** — "What will this property be worth in 5 years?"
- ⚠️ **Risk analysis** — "Is this area risky?"
- 📈 **ROI calculation** — "What is the expected ROI?"
- 🔄 **Property comparison** — "Compare Property A and B"

**Try asking:**
> "Should I buy a 3BHK in Bangalore for ₹1.2 Cr?"
> "Is a property in Hyderabad with crime index 70 risky?"
> "What will be the future price of a ₹80L property in Pune?"
    """

    def __init__(self, roi_engine=None, risk_engine=None,
                 valuation_engine=None, recommendation_engine=None,
                 forecasting_engine=None):
        self.roi_engine   = roi_engine
        self.risk_engine  = risk_engine
        self.val_engine   = valuation_engine
        self.rec_engine   = recommendation_engine
        self.fore_engine  = forecasting_engine
        self.history      = []

    def _detect_intent(self, text: str) -> str:
        text_lower = text.lower()
        for intent, keywords in self.INTENTS.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return "general"

    def _extract_price(self, text: str) -> float:
        """Extract price in INR from natural language."""
        # Patterns: ₹1.2 Cr, 80 lakhs, 1 crore, Rs 50L
        cr_match = re.search(r'(?:₹|rs\.?|rupees?)?\s*([\d.]+)\s*cr(?:ore)?s?',
                             text, re.I)
        l_match  = re.search(r'(?:₹|rs\.?|rupees?)?\s*([\d.]+)\s*(?:l|lakh|lac)s?',
                             text, re.I)
        if cr_match:
            return float(cr_match.group(1)) * 1e7
        elif l_match:
            return float(l_match.group(1)) * 1e5
        return None

    def _extract_city(self, text: str) -> str:
        cities = [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
            "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Noida",
        ]
        for city in cities:
            if city.lower() in text.lower():
                return city
        return "default"

    def _extract_bhk(self, text: str) -> int:
        m = re.search(r'(\d)\s*bhk', text, re.I)
        return int(m.group(1)) if m else 3

    # ── Response generators ───────────────────────────────────────────────────
    def _respond_invest(self, user_text: str) -> str:
        price = self._extract_price(user_text) or 8_000_000
        city  = self._extract_city(user_text)

        if self.roi_engine and self.risk_engine and self.rec_engine:
            roi_data = self.roi_engine.calculate(price, city)
            risk_data = self.risk_engine.calculate(
                crime_index=45, property_age=5, city=city,
                infrastructure_growth_score=60
            )
            rec = self.rec_engine.recommend(
                roi_data["primary_roi_pct"],
                risk_data["risk_score"],
                "fair_value",
                risk_data["risk_category"],
            )
            return (
                f"**Investment Analysis for {city} — ₹{price/1e5:.1f}L property**\n\n"
                f"📊 **5-Year Annualised ROI:** {roi_data['primary_roi_pct']}%\n"
                f"⚠️ **Risk Score:** {risk_data['risk_score']}/100 ({risk_data['risk_category']})\n"
                f"🎯 **Recommendation:** {rec['emoji']} **{rec['recommendation']}**\n\n"
                f"_{rec['reason']}_"
            )
        return (
            f"Based on typical market dynamics for {city}, "
            f"a property at ₹{price/1e5:.1f}L can be a good investment "
            "if the area has strong infrastructure growth and low crime. "
            "Consider getting a full analysis using the Investment Recommendation page."
        )

    def _respond_future_value(self, user_text: str) -> str:
        price = self._extract_price(user_text) or 8_000_000
        city  = self._extract_city(user_text)

        if self.fore_engine:
            fc = self.fore_engine.forecast(price, city)
            m = fc["milestones"]
            return (
                f"**Future Price Forecast — {city}**\n\n"
                f"Current Price: ₹{price/1e5:.1f}L\n"
                f"📅 1 Year:  ₹{m['1_year']['price']/1e5:.2f}L "
                f"({m['1_year']['total_roi_pct']:+.1f}%)\n"
                f"📅 3 Years: ₹{m['3_year']['price']/1e5:.2f}L "
                f"({m['3_year']['total_roi_pct']:+.1f}%)\n"
                f"📅 5 Years: ₹{m['5_year']['price']/1e5:.2f}L "
                f"({m['5_year']['total_roi_pct']:+.1f}%)\n\n"
                f"_(Based on {fc['growth_rate_pct']}% annual growth for {city})_"
            )
        return "I need the forecasting engine to compute future values. Please run the full training pipeline first."

    def _respond_risk(self, user_text: str) -> str:
        city = self._extract_city(user_text)
        ci_match = re.search(r'crime\s+(?:index\s+)?(?:of\s+)?(\d+)', user_text, re.I)
        ci = float(ci_match.group(1)) if ci_match else 45.0

        if self.risk_engine:
            risk = self.risk_engine.calculate(
                crime_index=ci, property_age=8, city=city,
                infrastructure_growth_score=55
            )
            return (
                f"**Risk Assessment — {city}**\n\n"
                f"{risk['risk_emoji']} **Risk Score:** {risk['risk_score']}/100\n"
                f"📂 **Category:** {risk['risk_category']}\n\n"
                f"Components:\n"
                + "\n".join([f"- {k}: {v:.1f}" for k, v in risk["components"].items()])
            )
        return f"Based on the crime index of {ci:.0f}, this area appears {'high' if ci > 65 else 'moderate' if ci > 35 else 'low'} risk."

    def _respond_roi(self, user_text: str) -> str:
        price = self._extract_price(user_text) or 8_000_000
        city  = self._extract_city(user_text)

        if self.roi_engine:
            roi = self.roi_engine.calculate(price, city)
            return (
                f"**ROI Analysis — {city}**\n\n"
                f"Current Price: ₹{price/1e5:.1f}L\n"
                f"📈 Annual Growth Rate: {roi['annual_growth_rate_pct']}%\n"
                f"🏠 Monthly Rental Estimate: ₹{roi['monthly_rental_estimate']:,.0f}\n"
                f"💹 Rental Yield: {roi['rental_yield_pct']}%\n"
                f"📊 5-Year Annualised ROI: {roi['primary_roi_pct']}%\n"
                f"💰 1-Year Projected Price: ₹{roi['projections']['1_year']['future_price']/1e5:.2f}L"
            )
        return f"A property in {city} typically sees 7-9% annual appreciation plus 3-4% rental yield."

    def _respond_overpriced(self, user_text: str) -> str:
        price = self._extract_price(user_text) or 8_000_000
        city  = self._extract_city(user_text)
        return (
            f"To determine if a ₹{price/1e5:.1f}L property in {city} is overpriced, "
            "I compare it against the predicted fair market value. "
            "Please use **Page 4: Investment Recommendation** or **Page 3: Price Prediction** "
            "to enter the full property details and get an accurate valuation verdict."
        )

    def _respond_general(self, user_text: str) -> str:
        return (
            "I'm not quite sure what you're asking. Here are some things I can help with:\n\n"
            "- 💰 Investment analysis\n"
            "- 🔮 Future price forecast\n"
            "- ⚠️ Risk assessment\n"
            "- 📈 ROI calculation\n\n"
            "Try rephrasing or type **help** for examples."
        )

    # ── Main respond method ───────────────────────────────────────────────────
    def respond(self, user_text: str) -> str:
        intent = self._detect_intent(user_text)
        log.info("Intent detected: %s", intent)

        if intent == "greeting":
            resp = "👋 Hello! I'm your AI Real Estate Investment Assistant. How can I help you today? Type **help** to see what I can do."
        elif intent == "help":
            resp = self.HELP_MSG
        elif intent == "should_invest":
            resp = self._respond_invest(user_text)
        elif intent == "future_value":
            resp = self._respond_future_value(user_text)
        elif intent == "risk":
            resp = self._respond_risk(user_text)
        elif intent == "roi":
            resp = self._respond_roi(user_text)
        elif intent == "overpriced":
            resp = self._respond_overpriced(user_text)
        else:
            resp = self._respond_general(user_text)

        # Log to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_text,
            "assistant": resp,
            "intent": intent,
        })
        return resp
