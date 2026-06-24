"""
AI-Powered Real Estate Intelligence Platform
Main Streamlit Application Entry Point
Run: streamlit run app.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import streamlit as st

st.set_page_config(
    page_title="🏠 AI Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0f172a; color: #e2e8f0; }
[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] [role="radiogroup"] p { color: #cbd5e1 !important; }
[data-testid="stSidebar"] [role="radiogroup"] span { color: #cbd5e1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #a78bfa);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(108,99,255,0.4); }
.stMetric { background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab"] { background: #1e293b; color: #94a3b8; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: #334155 !important; color: #a78bfa !important; }
.stSelectbox > div, .stNumberInput > div { background: #1e293b !important; }
div[data-testid="stDataFrame"] { background: #1e293b; border-radius: 12px; }
.stMarkdown h2 { color: #a78bfa; border-bottom: 2px solid #334155; padding-bottom: 8px; }
.stMarkdown h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:20px 0 10px">
  <div style="font-size:2.5rem">🏠</div>
  <div style="color:#a78bfa;font-weight:800;font-size:1.1rem">AI Real Estate</div>
  <div style="color:#64748b;font-size:.75rem">Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGE_MAP = {
    "🏠 Project Overview":          "pg1",
    "📊 EDA Dashboard":             "pg2",
    "🏷️ Price Prediction":          "pg3",
    "💼 Investment Recommendation":  "pg4",
    "⚠️ Risk Analysis":             "pg5",
    "🔮 Price Forecast":            "pg6",
    "🧠 Explainable AI":            "pg7",
    "⚖️ Property Comparison":       "pg8",
    "🌍 Market Insights":           "pg9",
    "🤖 AI Chat Assistant":         "pg10",
}

selection = st.sidebar.radio("Navigate", list(PAGE_MAP.keys()), label_visibility="collapsed")
page_key  = PAGE_MAP[selection]

# ── Load models/data lazily ───────────────────────────────────────────────────
from dashboard.loader import load_models, load_data, load_features, models_ready, data_ready

model = scaler = encoders = feature_cols = df = df_feat = None

if models_ready():
    try:
        model, scaler, encoders, feature_cols = load_models()
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
else:
    st.sidebar.warning("⚠️ Models not found.\nRun: `python pipeline.py`")

if data_ready():
    try:
        df      = load_data()
        df_feat = load_features()
    except Exception as e:
        st.sidebar.error(f"Data load error: {e}")
else:
    st.sidebar.info("ℹ️ Dataset not found.\nRun: `python pipeline.py`")

# ── Sidebar status ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**📟 System Status**")
st.sidebar.markdown(f"{'✅' if model else '❌'} ML Model")
st.sidebar.markdown(f"{'✅' if df is not None else '❌'} Dataset")
st.sidebar.markdown(f"{'✅' if df_feat is not None else '❌'} Features")
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='color:#64748b;font-size:.75rem;text-align:center'>v1.0 · Built with Streamlit · 2025</div>", unsafe_allow_html=True)

# ── Page Routing ──────────────────────────────────────────────────────────────
if page_key == "pg1":
    from dashboard.pages.pg1_overview import render
    render()

elif page_key == "pg2":
    from dashboard.pages.pg2_eda import render
    render(df)

elif page_key == "pg3":
    from dashboard.pages.pg3_predict import render
    render(model, scaler, encoders, feature_cols)

elif page_key == "pg4":
    from dashboard.pages.pg4_investment import render
    render(model)

elif page_key == "pg5":
    from dashboard.pages.pg5_risk import render
    render()

elif page_key == "pg6":
    from dashboard.pages.pg6_forecast import render
    render()

elif page_key == "pg7":
    from dashboard.pages.pg7_xai import render
    render(model, feature_cols, df_feat)

elif page_key == "pg8":
    from dashboard.pages.pg8_compare import render
    render()

elif page_key == "pg9":
    from dashboard.pages.pg9_market import render
    render(df)

elif page_key == "pg10":
    from dashboard.pages.pg10_chatbot import render
    render()
