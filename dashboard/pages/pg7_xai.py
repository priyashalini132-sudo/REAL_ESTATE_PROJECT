import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.explainability.shap_explainer import ExplainabilityEngine

def render(model, feature_cols, df_features: pd.DataFrame):
    st.markdown("## 🧠 Explainable AI Dashboard (SHAP)")
    st.info("Understand *why* the model predicts what it predicts — globally and for individual properties.")

    if model is None or df_features is None:
        st.error("Models/data not available. Run `python pipeline.py` first."); return

    tab1, tab2 = st.tabs(["🌐 Global Explainability", "🔍 Local Explainability"])

    with tab1:
        if st.button("⚡ Compute Global SHAP Values", key="shap_global_btn"):
            with st.spinner("Computing SHAP values (this may take ~30s)…"):
                from config.settings import TARGET_COLUMN
                X = df_features.drop(columns=[TARGET_COLUMN], errors="ignore")
                X = X[feature_cols] if feature_cols else X
                eng = ExplainabilityEngine(model=model, feature_names=list(X.columns))
                eng.fit(X)
                eng.compute_shap_values(X, max_samples=300)
                st.session_state["shap_eng"] = eng
            st.success("SHAP values computed!")

        if "shap_eng" in st.session_state:
            eng = st.session_state["shap_eng"]
            st.subheader("📊 Feature Importance (Mean |SHAP|)")
            fig = eng.plot_feature_importance(top_n=20)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Top Features Table")
            st.dataframe(eng.feature_importance_df().head(20), use_container_width=True)
        else:
            st.info("Click the button above to compute SHAP values.")

    with tab2:
        st.subheader("🔍 Local Explanation — Single Property")
        st.info("Select a sample index to explain one prediction.")

        if df_features is None:
            st.warning("Data not loaded."); return

        from config.settings import TARGET_COLUMN
        X = df_features.drop(columns=[TARGET_COLUMN], errors="ignore")
        if feature_cols:
            for c in feature_cols:
                if c not in X.columns: X[c] = 0
            X = X[feature_cols]

        idx = st.slider("Sample Index", 0, min(len(X)-1, 500), 0, key="shap_idx")
        row = X.iloc[[idx]]

        if st.button("🔬 Explain This Prediction", key="shap_local_btn"):
            with st.spinner("Computing local SHAP…"):
                if "shap_eng" not in st.session_state:
                    eng = ExplainabilityEngine(model=model, feature_names=list(X.columns))
                    eng.fit(X)
                    st.session_state["shap_eng"] = eng
                eng = st.session_state["shap_eng"]
                local = eng.local_explanation(row)

            pred_price = local["predicted"]
            st.metric("🤖 Predicted Price", f"₹{pred_price/1e5:.2f}L")
            st.metric("📊 SHAP Base Value", f"₹{local['base_value']/1e5:.2f}L")

            fig = eng.plot_waterfall(local, top_n=15)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 SHAP Values Table")
            st.dataframe(local["shap_df"].head(15), use_container_width=True)
