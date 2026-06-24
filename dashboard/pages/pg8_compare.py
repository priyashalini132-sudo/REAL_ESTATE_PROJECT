import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.roi.roi_engine import ROIEngine
from src.risk.risk_engine import RiskEngine
from src.valuation.valuation_engine import ValuationEngine, get_valuation_label_color
from src.recommendation.recommendation_engine import RecommendationEngine

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Noida"]

def _property_form(label, key_prefix):
    st.markdown(f"#### 🏠 {label}")
    c1, c2 = st.columns(2)
    city  = c1.selectbox("City", CITIES, key=f"{key_prefix}_city")
    price = c2.number_input("Listing Price (₹)", 500000, 100000000, 8000000, step=100000, key=f"{key_prefix}_price")
    c3, c4 = st.columns(2)
    pred  = c3.number_input("AI Predicted Price (₹)", 500000, 100000000, 8500000, step=100000, key=f"{key_prefix}_pred")
    age   = c4.number_input("Property Age (yrs)", 0, 50, 5, key=f"{key_prefix}_age")
    c5, c6 = st.columns(2)
    crime = c5.slider("Crime Index", 0, 100, 45, key=f"{key_prefix}_crime")
    infra = c6.slider("Infra Score",  0, 100, 60, key=f"{key_prefix}_infra")
    return city, price, pred, age, crime, infra

def _analyse(city, price, pred, age, crime, infra):
    roi_eng = ROIEngine(); risk_eng = RiskEngine()
    val_eng = ValuationEngine(); rec_eng = RecommendationEngine()
    roi  = roi_eng.calculate(price, city)
    risk = risk_eng.calculate(crime, age, city, infra)
    val  = val_eng.classify(price, pred)
    rec  = rec_eng.recommend(roi["primary_roi_pct"], risk["risk_score"],
                              val["status"], risk["risk_category"])
    return roi, risk, val, rec

def render():
    st.markdown("## ⚖️ Property Comparison Tool")
    st.info("Compare two properties side-by-side across all investment dimensions.")

    colA, colB = st.columns(2)
    with colA:
        cityA, priceA, predA, ageA, crimeA, infraA = _property_form("Property A", "cmp_a")
    with colB:
        cityB, priceB, predB, ageB, crimeB, infraB = _property_form("Property B", "cmp_b")

    if st.button("🔄 Compare Properties", type="primary", use_container_width=True, key="cmp_btn"):
        roiA, riskA, valA, recA = _analyse(cityA, priceA, predA, ageA, crimeA, infraA)
        roiB, riskB, valB, recB = _analyse(cityB, priceB, predB, ageB, crimeB, infraB)

        lblA, _, _ = get_valuation_label_color(valA["status"])
        lblB, _, _ = get_valuation_label_color(valB["status"])

        st.markdown("### 📊 Head-to-Head Summary")
        metrics = [
            ("💰 Listing Price", f"₹{priceA/1e5:.1f}L", f"₹{priceB/1e5:.1f}L"),
            ("🤖 AI Fair Value",  f"₹{valA['fair_value']/1e5:.1f}L", f"₹{valB['fair_value']/1e5:.1f}L"),
            ("🏷️ Valuation",      lblA, lblB),
            ("📈 Primary ROI",    f"{roiA['primary_roi_pct']}%", f"{roiB['primary_roi_pct']}%"),
            ("⚠️ Risk Score",     f"{riskA['risk_score']}/100", f"{riskB['risk_score']}/100"),
            ("🎯 Recommendation", f"{recA['emoji']} {recA['recommendation']}", f"{recB['emoji']} {recB['recommendation']}"),
            ("🏠 Rental/month",   f"₹{roiA['monthly_rental_estimate']:,.0f}", f"₹{roiB['monthly_rental_estimate']:,.0f}"),
        ]
        header = st.columns([3,3,3])
        header[0].markdown("**Metric**"); header[1].markdown("**Property A**"); header[2].markdown("**Property B**")
        for metric, vA, vB in metrics:
            r = st.columns([3,3,3])
            r[0].markdown(metric); r[1].markdown(f"`{vA}`"); r[2].markdown(f"`{vB}`")

        # Radar comparison
        st.markdown("### 🕸️ Radar Comparison")
        categories = ["ROI Score","Safety","Valuation","Rental Yield","Growth"]
        def norm_roi(v): return min(v / 25 * 100, 100)
        def norm_safe(v): return 100 - v
        def norm_val(s): return {"undervalued":100,"fair_value":60,"overvalued":20}.get(s,40)
        def norm_ry(v): return min(v / 5 * 100, 100)
        def norm_gr(v): return min(v / 10 * 100, 100)

        def scores(roi, risk, val):
            return [norm_roi(roi["primary_roi_pct"]), norm_safe(risk["risk_score"]),
                    norm_val(val["status"]), norm_ry(roi["rental_yield_pct"]),
                    norm_gr(roi["annual_growth_rate_pct"])]

        sA = scores(roiA, riskA, valA); sB = scores(roiB, riskB, valB)
        fig = go.Figure()
        for name, vals, color in [("Property A", sA, "#a78bfa"), ("Property B", sB, "#34d399")]:
            v = vals + [vals[0]]; c = categories + [categories[0]]
            # Convert hex to rgba to avoid Plotly Scatterpolar ValueError
            hex_c = color.lstrip("#")
            r_c, g_c, b_c = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            rgba_color = f"rgba({r_c}, {g_c}, {b_c}, 0.27)"
            fig.add_trace(go.Scatterpolar(r=v, theta=c, fill="toself",
                                          name=name, line_color=color, fillcolor=rgba_color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                          paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                          title="Investment Profile Comparison", height=450)
        st.plotly_chart(fig, use_container_width=True)
