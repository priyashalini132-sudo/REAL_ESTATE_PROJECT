import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.risk.risk_engine import RiskEngine

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Noida"]

def render():
    st.markdown("## ⚠️ Risk Analysis Engine")
    st.info("Compute a composite risk score using crime, property age, city volatility, infrastructure, and demand trend.")

    c1, c2, c3 = st.columns(3)
    city  = c1.selectbox("🏙️ City", CITIES, key="risk_city")
    crime = c2.slider("🚔 Crime Index (0-100)", 0, 100, 45, key="risk_crime")
    age   = c3.number_input("📅 Property Age (yrs)", 0, 50, 8, key="risk_age")
    c4, c5, c6 = st.columns(3)
    infra = c4.slider("🏗️ Infra Growth Score", 0, 100, 60, key="risk_infra")
    ppsf  = c5.number_input("💰 Price/sqft (₹)", 1000, 50000, 8000, step=500, key="risk_ppsf")
    city_med = c6.number_input("📊 City Median ₹/sqft", 1000, 50000, 8000, step=500, key="risk_med")

    if st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True, key="risk_btn"):
        eng = RiskEngine()
        result = eng.calculate(crime, age, city, infra, ppsf, city_med)
        score = result["risk_score"]
        cat   = result["risk_category"]
        color = result["risk_color"]

        st.markdown(f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:16px;
                    padding:30px;text-align:center;margin:16px 0">
          <div style="font-size:3rem">{result['risk_emoji']}</div>
          <div style="color:{color};font-size:2.4rem;font-weight:800">{cat}</div>
          <div style="color:#e2e8f0;font-size:1.5rem">Score: {score} / 100</div>
        </div>""", unsafe_allow_html=True)

        # Radar chart of components
        comps = result["components"]
        cats  = list(comps.keys())
        vals  = list(comps.values())
        cats.append(cats[0]); vals.append(vals[0])
        # Convert hex to rgba to avoid Plotly Scatterpolar ValueError
        hex_c = color.lstrip("#")
        r_c, g_c, b_c = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        rgba_color = f"rgba({r_c}, {g_c}, {b_c}, 0.27)"
        fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                        line_color=color, fillcolor=rgba_color))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            title="🕸️ Risk Component Radar",
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Component Breakdown")
        for k, v in result["components"].items():
            w = result["weights"].get(k.replace("crime","crime_index").replace("age","property_age")
                                      .replace("volatility","market_volatility")
                                      .replace("infra","infrastructure_growth_score")
                                      .replace("demand","demand_trend"), 0.2)
            col_a, col_b, col_c = st.columns([3,4,2])
            col_a.markdown(f"**{k.title()}**")
            col_b.progress(int(v))
            col_c.markdown(f"`{v:.1f}` (w={w})")

        # City comparison
        st.subheader("🏙️ City Risk Comparison")
        all_cities = CITIES
        city_risks = []
        for c in all_cities:
            r = eng.calculate(45, 8, c, 55)
            city_risks.append({"City": c, "Risk Score": r["risk_score"], "Category": r["risk_category"]})
        import pandas as pd
        df_r = pd.DataFrame(city_risks).sort_values("Risk Score")
        figb = px.bar(df_r, x="City", y="Risk Score", color="Risk Score",
                      color_continuous_scale="RdYlGn_r", title="Risk Scores Across Cities")
        figb.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                           font=dict(color="#e2e8f0"), template="plotly_dark")
        st.plotly_chart(figb, use_container_width=True)
