import sys
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from chatbot.chatbot import RealEstateChatbot
from src.roi.roi_engine import ROIEngine
from src.risk.risk_engine import RiskEngine
from src.valuation.valuation_engine import ValuationEngine
from src.recommendation.recommendation_engine import RecommendationEngine
from src.forecasting.forecasting_engine import ForecastingEngine

@st.cache_resource
def get_chatbot():
    return RealEstateChatbot(
        roi_engine=ROIEngine(),
        risk_engine=RiskEngine(),
        valuation_engine=ValuationEngine(),
        recommendation_engine=RecommendationEngine(),
        forecasting_engine=ForecastingEngine(),
    )

SUGGESTIONS = [
    "Should I invest in a 3BHK in Bangalore for ₹1.2 Cr?",
    "Is a property in Mumbai overpriced at ₹2 Cr?",
    "What will be the future price of a ₹80L property in Pune in 5 years?",
    "Is Hyderabad a risky city to invest in?",
    "What is the expected ROI in Noida?",
    "Help",
]

def render():
    st.markdown("## 🤖 AI Real Estate Chat Assistant")
    st.info("Ask investment questions in plain English — powered by our ML engines.")

    bot = get_chatbot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your **AI Real Estate Investment Assistant**. Ask me anything about property investments!\n\nType **help** to see what I can do."
        })

    # Suggestion chips
    st.markdown("**💡 Quick Questions:**")
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        if cols[i % 3].button(sug[:45] + ("…" if len(sug) > 45 else ""),
                              key=f"sug_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": sug})
            resp = bot.respond(sug)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})

    # Chat display
    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],
                             avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about investments, risks, ROI, future prices…", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                resp = bot.respond(user_input)
            st.markdown(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})

    # Clear button
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
