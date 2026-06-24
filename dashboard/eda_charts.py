"""
Dashboard EDA visualisations — Phase 4
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")
FONT = dict(color="#e2e8f0")


def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="price", nbins=80,
                       title="Price Distribution",
                       color_discrete_sequence=["#6C63FF"],
                       labels={"price": "Price (₹)"})
    fig.update_layout(**DARK, font=FONT)
    return fig


def area_vs_price(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(sample, x="area_sqft", y="price", color="city",
                     title="Area vs Price by City",
                     opacity=0.6,
                     labels={"area_sqft": "Area (sqft)", "price": "Price (₹)"})
    fig.update_layout(**DARK, font=FONT)
    return fig


def bhk_vs_price(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="bhk", y="price", color="bhk",
                 title="BHK vs Price Distribution",
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 labels={"bhk": "BHK", "price": "Price (₹)"})
    fig.update_layout(**DARK, font=FONT)
    return fig


def city_price_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(values="price", index="city",
                           columns="bhk", aggfunc="median") / 1e5
    fig = px.imshow(pivot, text_auto=".0f",
                    title="Median Price (₹L) by City & BHK",
                    color_continuous_scale="Viridis",
                    labels={"color": "Price (₹L)"})
    fig.update_layout(**DARK, font=FONT)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ["price", "area_sqft", "bhk", "bathrooms", "property_age",
                "metro_distance_km", "crime_index", "infrastructure_growth_score"]
    corr = df[[c for c in num_cols if c in df.columns]].corr()
    fig = px.imshow(corr, text_auto=".2f",
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1)
    fig.update_layout(**DARK, font=FONT)
    return fig


def location_price_bar(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("city")["price"].median().reset_index().sort_values("price", ascending=True)
    fig = px.bar(avg, x="price", y="city", orientation="h",
                 title="Median Property Price by City",
                 color="price", color_continuous_scale="Viridis",
                 labels={"price": "Median Price (₹)", "city": "City"})
    fig.update_layout(**DARK, font=FONT)
    return fig


def property_type_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["property_type"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index,
                 title="Property Type Distribution",
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 hole=0.4)
    fig.update_layout(**DARK, font=FONT)
    return fig


def furnished_price_violin(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(df, x="furnished_status", y="price",
                    color="furnished_status", box=True,
                    title="Price by Furnishing Status",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(**DARK, font=FONT)
    return fig


def price_per_sqft_by_city(df: pd.DataFrame) -> go.Figure:
    if "price_per_sqft" not in df.columns:
        df = df.copy()
        df["price_per_sqft"] = df["price"] / df["area_sqft"]
    fig = px.box(df, x="city", y="price_per_sqft", color="city",
                 title="Price per Sqft by City",
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(**DARK, font=FONT, xaxis_tickangle=-30)
    return fig
