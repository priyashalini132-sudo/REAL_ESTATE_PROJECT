import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.roi.roi_engine import ROIEngine
from src.risk.risk_engine import RiskEngine
from config.settings import ANNUAL_GROWTH_RATES

CITIES = list(ANNUAL_GROWTH_RATES.keys())[:-1]  # exclude 'default'
DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"))

def _city_insights():
    roi_eng = ROIEngine(); risk_eng = RiskEngine()
    rows = []
    for city in CITIES:
        price = 8_000_000
        roi  = roi_eng.calculate(price, city)
        risk = risk_eng.calculate(45, 8, city, 55)
        rows.append({
            "City": city,
            "Annual Growth %": roi["annual_growth_rate_pct"],
            "Primary ROI %":   roi["primary_roi_pct"],
            "Risk Score":      risk["risk_score"],
            "Risk Category":   risk["risk_category"],
            "Rental Yield %":  roi["rental_yield_pct"],
            "5yr Price (₹L)":  round(roi["future_value_5yr"]/1e5, 1),
        })
    return pd.DataFrame(rows)

def render(df: pd.DataFrame):
    st.markdown("## 🌍 Market Insights Dashboard")
    st.info("City-level investment attractiveness, growth trends, and risk landscape.")

    city_df = _city_insights()

    # KPI row
    best_roi  = city_df.loc[city_df["Primary ROI %"].idxmax()]
    best_safe = city_df.loc[city_df["Risk Score"].idxmin()]
    best_grow = city_df.loc[city_df["Annual Growth %"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Highest ROI City",    best_roi["City"],  f"{best_roi['Primary ROI %']}%")
    c2.metric("🛡️ Safest City",         best_safe["City"], f"Risk {best_safe['Risk Score']:.0f}/100")
    c3.metric("📈 Fastest Growing",     best_grow["City"], f"{best_grow['Annual Growth %']}% p.a.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 ROI & Growth","⚠️ Risk Map","🗺️ Top Investment Areas","🏆 Top 10 Properties"])
    
    with tab1:
        fig = px.bar(city_df.sort_values("Primary ROI %", ascending=False),
                     x="City", y="Primary ROI %", color="Annual Growth %",
                     color_continuous_scale="Viridis", title="📈 City ROI Ranking")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.scatter(city_df, x="Risk Score", y="Primary ROI %",
                          size="Annual Growth %", color="City", text="City",
                          title="⚖️ Risk vs ROI Bubble Chart",
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        fig2.update_traces(textposition="top center")
        fig2.update_layout(**DARK, height=450)
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab2:
        fig = px.bar(city_df.sort_values("Risk Score"),
                     x="City", y="Risk Score", color="Risk Score",
                     color_continuous_scale="RdYlGn_r", title="⚠️ City Risk Scores")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(city_df[["City","Risk Score","Risk Category"]].sort_values("Risk Score"),
                     use_container_width=True)
                     
    with tab3:
        city_df["Investment Score"] = (
            city_df["Primary ROI %"] / city_df["Primary ROI %"].max() * 40 +
            (100 - city_df["Risk Score"]) / 100 * 30 +
            city_df["Annual Growth %"] / city_df["Annual Growth %"].max() * 30
        ).round(1)
        ranked = city_df.sort_values("Investment Score", ascending=False)
        st.subheader("🏆 Top Investment Cities")
        for _, row in ranked.iterrows():
            score = row["Investment Score"]
            color = "#22c55e" if score > 65 else "#f59e0b" if score > 45 else "#ef4444"
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:6px;
                        border-left:4px solid {color};display:flex;justify-content:space-between">
              <span style="color:#f1f5f9;font-weight:600;font-size:1rem">🏙️ {row['City']}</span>
              <span style="color:{color};font-weight:700">Score: {score:.1f}</span>
              <span style="color:#94a3b8">ROI: {row['Primary ROI %']}% | Risk: {row['Risk Score']:.0f}</span>
            </div>""", unsafe_allow_html=True)
            
        # Locality insights from dataset
        if df is not None and "locality" in df.columns and "price_per_sqft" in df.columns:
            st.subheader("📍 Top Localities by Price/sqft")
            top = df.groupby(["city","locality"])["price_per_sqft"].median().reset_index()
            top = top.sort_values("price_per_sqft", ascending=False).head(20)
            fig3 = px.bar(top, x="price_per_sqft", y="locality", color="city",
                          orientation="h", title="Top 20 Localities — Median ₹/sqft")
            fig3.update_layout(**DARK, height=550)
            st.plotly_chart(fig3, use_container_width=True)
            
    with tab4:
        st.subheader("🏆 Top 10 Ranked Investment Properties")
        st.info("Properties ranked automatically by combining safety (crime rates), infrastructure growth, and pricing value relative to local city averages (undervalued metric).")
        
        if df is None or df.empty:
            st.warning("Processed dataset not found. Run pipeline.py first.")
        else:
            df_rank = df.copy()
            
            # Compute a safety score (100 - crime_index)
            safety = 100 - df_rank["crime_index"].fillna(45)
            # Growth score (0-100)
            growth = df_rank["infrastructure_growth_score"].fillna(60)
            
            # Valuation factor: ratio of property price/sqft vs city median
            city_meds = df_rank.groupby("city")["price_per_sqft"].transform("median")
            val_ratio = df_rank["price_per_sqft"] / city_meds
            # Lower ratio is better (undervalued). Score is high if ratio is low.
            val_score = 100 - (val_ratio * 50)
            val_score = np.clip(val_score, 0, 100)
            
            # Attractiveness Score
            df_rank["Attractiveness Score"] = (safety * 0.3 + growth * 0.3 + val_score * 0.4).round(1)
            
            # Filter and sort
            top_props = df_rank.sort_values("Attractiveness Score", ascending=False).head(10)
            
            # Format and show columns
            top_props_display = top_props[[
                "property_id", "city", "locality", "property_type", "bhk",
                "area_sqft", "price", "price_per_sqft", "Attractiveness Score"
            ]].copy()
            
            # Rename for display
            top_props_display.columns = [
                "ID", "City", "Locality", "Type", "BHK", 
                "Area (sqft)", "Price (₹)", "Price/sqft (₹)", "Attractiveness Score"
            ]
            
            st.dataframe(top_props_display.style.format({
                "Price (₹)": "₹{:,.0f}",
                "Price/sqft (₹)": "₹{:,.0f}"
            }), use_container_width=True)
            
            # Visualise ranking
            fig4 = px.bar(top_props_display.sort_values("Attractiveness Score"), x="Attractiveness Score", y="Locality", color="City",
                          orientation="h", title="Top 10 Investment Properties by Attractiveness Score",
                          hover_data=["Price (₹)", "Area (sqft)", "BHK"])
            fig4.update_layout(**DARK, height=400)
            st.plotly_chart(fig4, use_container_width=True)
