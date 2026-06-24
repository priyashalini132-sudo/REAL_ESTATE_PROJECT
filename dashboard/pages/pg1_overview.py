import streamlit as st

def render():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:16px;padding:40px;margin-bottom:24px;border:1px solid #334155">
      <h1 style="color:#a78bfa;margin:0;font-size:2.4rem">🏠 AI Real Estate Intelligence Platform</h1>
      <p style="color:#94a3b8;margin-top:8px;font-size:1.1rem">Production-Grade Property Valuation · Investment Analysis · Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, val in [
        (c1,"🏗️","Dataset","20,000+ Records"),
        (c2,"🤖","ML Models","6 Algorithms"),
        (c3,"🏙️","Cities","10 Metro Areas"),
        (c4,"📊","Features","25+ Attributes"),
    ]:
        col.markdown(f"""
        <div style="background:#1e293b;border-radius:12px;padding:20px;text-align:center;border:1px solid #334155">
          <div style="font-size:2rem">{icon}</div>
          <div style="color:#94a3b8;font-size:.8rem;margin-top:4px">{label}</div>
          <div style="color:#a78bfa;font-weight:700;font-size:1.1rem">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📐 System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────┐
    │              DATA LAYER                             │
    │  Synthetic 20K Dataset · 25 Features · SQLite DB   │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │              ML PIPELINE                            │
    │  Clean → Engineer → Train (XGB/LGB/CB/RF/GB/Ridge) │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │           INTELLIGENCE ENGINES                      │
    │  Valuation · ROI · Risk · Forecast · Ranking       │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │           STREAMLIT DASHBOARD (10 Pages)            │
    │  Overview · EDA · Predict · Invest · Risk ·        │
    │  Forecast · XAI · Compare · Market · Chatbot       │
    └─────────────────────────────────────────────────────┘
    ```
    """)

    st.subheader("🗺️ Platform Modules")
    modules = [
        ("Phase 2", "Data Generation", "20K synthetic Indian real-estate records with realistic pricing"),
        ("Phase 3", "Data Cleaning", "Missing-value imputation, outlier capping, consistency checks"),
        ("Phase 4", "EDA", "Price distributions, geo heatmaps, correlation analysis"),
        ("Phase 5", "Feature Engineering", "OHE + Label encoding, StandardScaler, tree-based selection"),
        ("Phase 6", "Model Training", "Ridge, RF, GB, XGBoost, LightGBM, CatBoost + CV + tuning"),
        ("Phase 7", "Valuation Engine", "Fair value / Undervalued / Overvalued classification"),
        ("Phase 8", "ROI Engine", "Capital appreciation + rental yield + CAGR projections"),
        ("Phase 9", "Risk Engine", "Crime, age, volatility, infra, demand → composite risk score"),
        ("Phase 10","Recommendation","Buy / Hold / Sell rules engine"),
        ("Phase 11","Forecasting","1 / 3 / 5-year price forecast with confidence bands"),
        ("Phase 12","Explainable AI","SHAP global & local explanations, waterfall plots"),
        ("Phase 13","Ranking Engine","Investment attractiveness scoring & ranking"),
        ("Phase 14","AI Chatbot","Intent-based NLP real-estate assistant"),
    ]
    for ph, name, desc in modules:
        st.markdown(f"""
        <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:6px;border-left:3px solid #a78bfa;display:flex;gap:12px">
          <span style="color:#a78bfa;font-weight:700;min-width:80px">{ph}</span>
          <span style="color:#f1f5f9;font-weight:600;min-width:160px">{name}</span>
          <span style="color:#94a3b8">{desc}</span>
        </div>""", unsafe_allow_html=True)
