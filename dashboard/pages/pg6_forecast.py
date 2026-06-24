import sys
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.forecasting.forecasting_engine import ForecastingEngine

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Noida"]

def render():
    st.markdown("## 🔮 Future Price Forecast")
    st.info("Predict 1-year, 3-year, and 5-year property prices using CAGR-based forecasting with confidence bands.")

    c1, c2, c3 = st.columns(3)
    city  = c1.selectbox("🏙️ City", CITIES, key="fc_city")
    price = c2.number_input("💰 Current Price (₹)", 500000, 100000000, 8000000, step=100000, key="fc_price")
    name  = c3.text_input("🏷️ Property Name", "My Property", key="fc_name")
    custom_gr = st.slider("📈 Custom Growth Rate % (0 = use city default)", 0.0, 20.0, 0.0, 0.5, key="fc_gr")

    if st.button("📊 Generate Forecast", type="primary", use_container_width=True, key="fc_btn"):
        eng = ForecastingEngine()
        gr  = (custom_gr / 100) if custom_gr > 0 else None
        fc  = eng.forecast(price, city, custom_growth_rate=gr)
        m   = fc["milestones"]

        col1, col2, col3 = st.columns(3)
        for col, yr in [(col1,"1_year"),(col2,"3_year"),(col3,"5_year")]:
            d = m[yr]
            lbl = yr.replace("_"," ").title()
            delta = f"+{d['total_roi_pct']}% ROI"
            col.metric(f"📅 {lbl}", f"₹{d['price']/1e5:.2f}L", delta)

        fig = eng.plot_forecast(fc, name)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detailed Projections")
        rows = []
        for yr_key, d in m.items():
            rows.append({
                "Horizon": yr_key.replace("_"," ").title(),
                "Projected Price": f"₹{d['price']/1e5:.2f}L",
                "Upper Bound": f"₹{d['price_upper']/1e5:.2f}L",
                "Lower Bound": f"₹{d['price_lower']/1e5:.2f}L",
                "Total Gain": f"₹{d['total_gain']/1e5:.2f}L",
                "ROI %": f"{d['total_roi_pct']}%",
            })
        st.dataframe(rows, use_container_width=True)

        st.markdown(f"**Growth rate used:** {fc['growth_rate_pct']}% p.a. for **{city}**")
