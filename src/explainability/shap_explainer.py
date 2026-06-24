"""
Phase 12 – Explainable AI Module (SHAP)
Global and local explainability with SHAP.
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import BEST_MODEL_PATH, FEATURE_COLUMNS_PATH

log = logging.getLogger(__name__)


class ExplainabilityEngine:
    """SHAP-based global and local explainer."""

    def __init__(self, model=None, feature_names: list = None):
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        self.shap_values = None
        self._background = None

    # ── Build explainer ────────────────────────────────────────────────────────
    def fit(self, X_background: pd.DataFrame, model=None):
        if model:
            self.model = model
        if self.feature_names == []:
            self.feature_names = list(X_background.columns)

        log.info("Building SHAP TreeExplainer …")
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to KernelExplainer for non-tree models
            bg = shap.sample(X_background, 100)
            self.explainer = shap.KernelExplainer(
                self.model.predict, bg
            )
        self._background = X_background
        return self

    # ── Global SHAP Values ─────────────────────────────────────────────────────
    def compute_shap_values(self, X: pd.DataFrame,
                             max_samples: int = 500):
        sample = X.sample(min(max_samples, len(X)), random_state=42)
        log.info("Computing SHAP values for %d samples …", len(sample))
        self.shap_values = self.explainer.shap_values(sample)
        self._shap_sample = sample
        return self.shap_values, sample

    # ── Feature Importance (SHAP) ─────────────────────────────────────────────
    def feature_importance_df(self) -> pd.DataFrame:
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self._shap_sample.columns.tolist(),
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def plot_feature_importance(self, top_n: int = 20) -> go.Figure:
        df = self.feature_importance_df().head(top_n)
        fig = go.Figure(go.Bar(
            x=df["importance"][::-1],
            y=df["feature"][::-1],
            orientation="h",
            marker=dict(
                color=df["importance"][::-1],
                colorscale="Viridis",
                showscale=True,
            )
        ))
        fig.update_layout(
            title="🔬 Global SHAP Feature Importance",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            height=600,
        )
        return fig

    # ── Local (single property) explanation ──────────────────────────────────
    def local_explanation(self, row: pd.DataFrame) -> dict:
        """SHAP values for a single prediction."""
        sv = self.explainer.shap_values(row)
        if hasattr(self.explainer, "expected_value"):
            base_val = float(self.explainer.expected_value)
        else:
            base_val = float(np.mean(self.model.predict(self._background)))

        shap_df = pd.DataFrame({
            "feature": row.columns.tolist(),
            "value": row.iloc[0].values,
            "shap_value": sv[0] if sv.ndim > 1 else sv,
        }).sort_values("shap_value", key=abs, ascending=False)

        return {
            "base_value": base_val,
            "shap_df": shap_df,
            "predicted": base_val + shap_df["shap_value"].sum(),
        }

    def plot_waterfall(self, local_exp: dict,
                       top_n: int = 15) -> go.Figure:
        df = local_exp["shap_df"].head(top_n)
        colors = ["#22c55e" if v >= 0 else "#ef4444"
                  for v in df["shap_value"]]
        fig = go.Figure(go.Bar(
            x=df["shap_value"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.0f}" for v in df["shap_value"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="💧 SHAP Waterfall – Local Explanation",
            xaxis_title="SHAP Value (impact on prediction)",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            height=500,
        )
        return fig
