import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"))

def render(df: pd.DataFrame):
    st.markdown("## 📊 Exploratory Data Analysis Dashboard")
    if df is None or df.empty:
        st.warning("Run the pipeline first: `python pipeline.py`"); return

    tab1, tab2, tab3, tab4 = st.tabs(["Univariate","Bivariate","Multivariate","Location"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="price", nbins=80, title="💰 Price Distribution",
                               color_discrete_sequence=["#a78bfa"])
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="area_sqft", nbins=80, title="📐 Area (sqft) Distribution",
                               color_discrete_sequence=["#34d399"])
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            vc = df["city"].value_counts().reset_index()
            fig = px.bar(vc, x="city", y="count", title="🏙️ Properties by City",
                         color="count", color_continuous_scale="Viridis")
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            vc2 = df["property_type"].value_counts().reset_index()
            fig = px.pie(vc2, names="property_type", values="count",
                         title="🏘️ Property Type Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                             x="area_sqft", y="price", color="city",
                             title="📐 Area vs Price", opacity=0.6)
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            med = df.groupby("bhk")["price"].median().reset_index()
            fig = px.bar(med, x="bhk", y="price", title="🛏️ BHK vs Median Price",
                         color="price", color_continuous_scale="Plasma")
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            med2 = df.groupby("city")["price"].median().reset_index().sort_values("price", ascending=False)
            fig = px.bar(med2, x="city", y="price", title="🏙️ City vs Median Price",
                         color="price", color_continuous_scale="Viridis")
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.box(df, x="furnished_status", y="price", color="furnished_status",
                         title="🛋️ Furnished Status vs Price",
                         color_discrete_sequence=["#a78bfa","#34d399","#f59e0b"])
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = ["price","area_sqft","bhk","bathrooms","property_age",
                    "crime_index","infrastructure_growth_score","metro_distance_km"]
        available = [c for c in num_cols if c in df.columns]
        corr = df[available].corr()
        fig = px.imshow(corr, text_auto=".2f", title="🔥 Correlation Heatmap",
                        color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(**DARK, height=500)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.scatter(df.sample(min(2000, len(df)), random_state=1),
                              x="crime_index", y="price", color="city",
                              title="🚔 Crime Index vs Price", opacity=0.6)
            fig2.update_layout(**DARK)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.scatter(df.sample(min(2000, len(df)), random_state=2),
                              x="infrastructure_growth_score", y="price", color="property_type",
                              title="🏗️ Infra Score vs Price", opacity=0.6)
            fig3.update_layout(**DARK)
            st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        med_city = df.groupby("city").agg(
            median_price=("price","median"),
            count=("price","count"),
            avg_infra=("infrastructure_growth_score","mean")
        ).reset_index()
        fig = px.scatter(med_city, x="city", y="median_price", size="count",
                         color="avg_infra", title="🗺️ City: Median Price & Infra Score",
                         color_continuous_scale="Plasma")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, use_container_width=True)

        if "price_per_sqft" in df.columns:
            top_loc = df.groupby("locality")["price_per_sqft"].median().sort_values(ascending=False).head(20)
            fig2 = px.bar(top_loc.reset_index(), x="price_per_sqft", y="locality",
                          orientation="h", title="💎 Top 20 Localities by Price/sqft",
                          color="price_per_sqft", color_continuous_scale="Viridis")
            fig2.update_layout(**DARK, height=550)
            st.plotly_chart(fig2, use_container_width=True)
