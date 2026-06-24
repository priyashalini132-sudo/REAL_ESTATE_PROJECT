# 🏠 AI-Powered Real Estate Intelligence Platform

> Production-grade property valuation, investment analysis & market intelligence — built with Python, Streamlit, XGBoost, SHAP, and Plotly.

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to the project
cd real_estate_platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (generates data, trains models — ~5 min)
python pipeline.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
real_estate_platform/
│
├── app.py                        # Streamlit entry point (10 pages)
├── pipeline.py                   # End-to-end setup runner
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py               # All global constants & paths
│
├── data/
│   ├── raw/real_estate_raw.csv   # Generated dataset (20K rows)
│   └── processed/                # Cleaned + engineered features
│
├── models/saved/
│   ├── best_model.pkl            # Best trained model
│   ├── scaler.pkl                # StandardScaler
│   ├── encoders.pkl              # Label encoders
│   ├── feature_columns.pkl       # Selected feature list
│   └── model_comparison.json     # All model metrics
│
├── src/
│   ├── preprocessing/
│   │   ├── data_generator.py     # Phase 2 – synthetic dataset
│   │   ├── data_cleaner.py       # Phase 3 – cleaning pipeline
│   │   └── feature_engineer.py  # Phase 5 – encoding & scaling
│   ├── training/
│   │   └── model_trainer.py      # Phase 6 – 6 models + CV + tuning
│   ├── valuation/
│   │   └── valuation_engine.py   # Phase 7 – fair/over/undervalued
│   ├── roi/
│   │   └── roi_engine.py         # Phase 8 – ROI & rental yield
│   ├── risk/
│   │   └── risk_engine.py        # Phase 9 – composite risk score
│   ├── recommendation/
│   │   └── recommendation_engine.py  # Phase 10 – Buy/Hold/Sell
│   ├── forecasting/
│   │   └── forecasting_engine.py # Phase 11 – 1/3/5-year forecast
│   ├── explainability/
│   │   └── shap_explainer.py     # Phase 12 – SHAP global + local
│   └── ranking/
│       └── ranking_engine.py     # Phase 13 – property ranking
│
├── chatbot/
│   └── chatbot.py                # Phase 14 – AI investment assistant
│
├── dashboard/
│   ├── loader.py                 # Cached model/data loading
│   └── pages/
│       ├── pg1_overview.py       # Project overview & architecture
│       ├── pg2_eda.py            # Exploratory data analysis
│       ├── pg3_predict.py        # Price prediction form
│       ├── pg4_investment.py     # Investment recommendation
│       ├── pg5_risk.py           # Risk analysis & radar chart
│       ├── pg6_forecast.py       # Future price forecast
│       ├── pg7_xai.py            # Explainable AI (SHAP)
│       ├── pg8_compare.py        # Property A vs B comparison
│       ├── pg9_market.py         # Market insights & rankings
│       └── pg10_chatbot.py       # AI chat assistant
│
├── database/
│   └── db_manager.py             # SQLite manager (PostgreSQL-ready)
│
└── reports/                      # Auto-generated JSON reports
```

---

## 🧠 ML Models Trained

| Model | Description |
|-------|-------------|
| Ridge Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting |
| XGBoost | Optimised gradient boosting |
| LightGBM | Fast gradient boosting |
| CatBoost | Categorical-aware boosting |

Best model selected by R² with 5-fold cross-validation. XGBoost is further hyperparameter-tuned via RandomizedSearchCV.

---

## 📊 Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | Overview | System architecture & module breakdown |
| 2 | EDA | Univariate, bivariate, multivariate analysis |
| 3 | Price Prediction | AI price prediction form |
| 4 | Investment | Buy/Hold/Sell recommendation + ROI gauges |
| 5 | Risk Analysis | Composite risk score + radar chart |
| 6 | Forecast | 1/3/5-year price forecast with confidence bands |
| 7 | Explainable AI | SHAP global & local explanations |
| 8 | Comparison | Side-by-side property comparison |
| 9 | Market Insights | City rankings, risk map, top localities |
| 10 | AI Chatbot | Natural language investment assistant |

---

## 🏗️ Dataset

- **Size:** 20,000+ synthetic records (statistically realistic)
- **Cities:** Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, Kolkata, Ahmedabad, Jaipur, Noida
- **Features:** 25 attributes spanning property specs, location, distances, and economic indicators
- **Target:** `price` (₹ INR)

---

## 🚀 Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Add `requirements.txt`
5. Note: Run `pipeline.py` locally first and commit generated model files

---

## 📄 License

MIT License — free for commercial and academic use.
