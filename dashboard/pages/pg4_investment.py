import sys
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.roi.roi_engine import ROIEngine
from src.risk.risk_engine import RiskEngine
from src.valuation.valuation_engine import ValuationEngine, get_valuation_label_color
from src.recommendation.recommendation_engine import RecommendationEngine

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Noida"]

def _gauge(title, value, max_val=100, color="#a78bfa"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e2e8f0", "size": 14}},
        gauge={"axis": {"range": [0, max_val], "tickcolor": "#94a3b8"},
               "bar": {"color": color},
               "bgcolor": "#1e293b",
               "bordercolor": "#334155",
               "steps": [{"range":[0,max_val*0.4],"color":"#1e293b"},
                         {"range":[max_val*0.4,max_val*0.7],"color":"#1e293b"}]},
        number={"font": {"color": "#e2e8f0"}},
    ))
    fig.update_layout(paper_bgcolor="#0f172a", height=220, margin=dict(t=40,b=10,l=10,r=10))
    return fig

def render(model):
    st.markdown("## 💼 Investment Recommendation Engine")
    st.info("Enter a property's details to get a **Buy / Hold / Sell** recommendation backed by AI.")

    c1, c2, c3 = st.columns(3)
    city         = c1.selectbox("🏙️ City", CITIES, key="inv_city")
    current_price= c2.number_input("💰 Listing Price (₹)", 500000, 100000000, 8000000, step=100000, key="inv_price")
    pred_price   = c3.number_input("🤖 AI Predicted Price (₹)", 500000, 100000000, 8500000, step=100000, key="inv_pred")

    c4, c5, c6 = st.columns(3)
    age    = c4.number_input("📅 Property Age (yrs)", 0, 50, 5, key="inv_age")
    crime  = c5.slider("🚔 Crime Index", 0, 100, 40, key="inv_crime")
    infra  = c6.slider("🏗️ Infra Score", 0, 100, 65, key="inv_infra")

    if st.button("🎯 Generate Recommendation", type="primary", use_container_width=True, key="inv_btn"):
        roi_eng  = ROIEngine()
        risk_eng = RiskEngine()
        val_eng  = ValuationEngine()
        rec_eng  = RecommendationEngine()

        roi_data  = roi_eng.calculate(current_price, city)
        risk_data = risk_eng.calculate(crime, age, city, infra)
        val_data  = val_eng.classify(current_price, pred_price)
        rec       = rec_eng.recommend(roi_data["primary_roi_pct"],
                                      risk_data["risk_score"],
                                      val_data["status"],
                                      risk_data["risk_category"])

        label, emoji, color = get_valuation_label_color(val_data["status"])
        action = rec["recommendation"]
        act_color = rec["color"]

        # Big recommendation badge
        st.markdown(f"""
        <div style="background:{act_color}22;border:2px solid {act_color};border-radius:16px;
                    padding:30px;text-align:center;margin:16px 0">
          <div style="font-size:3rem">{rec['emoji']}</div>
          <div style="color:{act_color};font-size:2.5rem;font-weight:800">{action}</div>
          <div style="color:#e2e8f0;margin-top:8px">{rec['reason']}</div>
          <div style="color:#94a3b8;margin-top:4px">Confidence: {rec['confidence_pct']}%</div>
        </div>""", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("📈 ROI (5yr Ann.)", f"{roi_data['primary_roi_pct']}%")
        mc2.metric("⚠️ Risk Score",    f"{risk_data['risk_score']}/100")
        mc3.metric("🏷️ Valuation",     label)
        mc4.metric("💹 Annual Growth", f"{roi_data['annual_growth_rate_pct']}%")

        col1, col2 = st.columns(2)
        col1.plotly_chart(_gauge("ROI %", roi_data["primary_roi_pct"], 30, "#22c55e"), use_container_width=True)
        col2.plotly_chart(_gauge("Risk Score", risk_data["risk_score"], 100, "#ef4444"), use_container_width=True)

        st.subheader("📊 Year-by-Year Projections")
        rows = []
        for yr, d in roi_data["projections"].items():
            rows.append({"Year": yr, "Future Price": f"₹{d['future_price']/1e5:.2f}L",
                         "Capital Gain": f"₹{d['capital_gain']/1e5:.2f}L",
                         "Total Return %": f"{d['total_return_pct']}%"})
        st.dataframe(rows, use_container_width=True)
