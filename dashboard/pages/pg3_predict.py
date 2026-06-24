import sys, pickle
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from config.settings import (CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
                              SCALER_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH)
from src.roi.roi_engine import ROIEngine
from src.risk.risk_engine import RiskEngine
from src.valuation.valuation_engine import ValuationEngine
from src.recommendation.recommendation_engine import RecommendationEngine

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Noida"]
PROP_TYPES = ["Apartment","Villa","Independent House","Plot","Penthouse"]
FURNISHED = ["Fully Furnished","Semi Furnished","Unfurnished"]
LOCALITIES = {
    "Mumbai":["Bandra","Andheri","Worli","Powai","Juhu"],
    "Delhi":["Dwarka","Rohini","Saket","Vasant Kunj","Lajpat Nagar"],
    "Bangalore":["Koramangala","Whitefield","Indiranagar","HSR Layout","Electronic City"],
    "Hyderabad":["Gachibowli","HITEC City","Banjara Hills","Jubilee Hills","Madhapur"],
    "Chennai":["Anna Nagar","Velachery","OMR","Porur","Adyar"],
    "Pune":["Hinjewadi","Kothrud","Wakad","Viman Nagar","Baner"],
    "Kolkata":["Salt Lake","New Town","Behala","Howrah","Rajarhat"],
    "Ahmedabad":["SG Highway","Bopal","Prahlad Nagar","Satellite","Maninagar"],
    "Jaipur":["Malviya Nagar","C-Scheme","Vaishali Nagar","Mansarovar","Jagatpura"],
    "Noida":["Sector 18","Sector 62","Sector 137","Greater Noida","Sector 50"],
}

def _build_input_row(city, locality, area, prop_type, bhk, bathrooms, balcony,
                     floor, total_floors, age, furnished, parking,
                     hosp, school, metro, mall, airport, pop_density, crime, infra):
    return pd.DataFrame([{
        "property_type": prop_type, "bhk": bhk, "bathrooms": bathrooms,
        "balcony": balcony, "floor_number": floor, "total_floors": total_floors,
        "area_sqft": area, "property_age": age, "furnished_status": furnished,
        "parking_availability": parking, "city": city, "area": "Central",
        "locality": locality, "pincode": 400001,
        "hospital_distance_km": hosp, "school_distance_km": school,
        "metro_distance_km": metro, "mall_distance_km": mall,
        "airport_distance_km": airport, "population_density": pop_density,
        "crime_index": crime, "infrastructure_growth_score": infra,
    }])

def _predict(row_df, model, scaler, encoders, feature_cols):
    df = row_df.copy()
    # OHE
    ohe_cols = ["property_type","furnished_status","parking_availability"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
    # LE
    for col in ["city","area","locality"]:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = 0
    # drop unwanted
    df = df.drop(columns=["property_id","pincode","price_per_sqft","price"], errors="ignore")
    # scale
    num_present = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_present] = scaler.transform(df[num_present])
    # align columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return float(model.predict(df)[0])

def _predict_bulk(df_input, model, scaler, encoders, feature_cols):
    defaults = {
        "property_type": "Apartment", "bhk": 3, "bathrooms": 2, "balcony": 1,
        "floor_number": 3, "total_floors": 10, "area_sqft": 1200, "property_age": 5,
        "furnished_status": "Semi Furnished", "parking_availability": "Yes",
        "city": "Mumbai", "area": "Central", "locality": "Bandra", "pincode": 400001,
        "hospital_distance_km": 2.0, "school_distance_km": 1.5, "metro_distance_km": 3.0,
        "mall_distance_km": 4.0, "airport_distance_km": 20.0, "population_density": 8000,
        "crime_index": 45, "infrastructure_growth_score": 60
    }
    df = df_input.copy()
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    preds = []
    for idx, row in df.iterrows():
        row_df = pd.DataFrame([row.to_dict()])
        try:
            val = _predict(row_df, model, scaler, encoders, feature_cols)
            preds.append(val)
        except Exception:
            preds.append(np.nan)
    return preds

def analyze_bulk_rows(df_input, model, scaler, encoders, feature_cols):
    prices = _predict_bulk(df_input, model, scaler, encoders, feature_cols)
    roi_eng = ROIEngine()
    risk_eng = RiskEngine()
    val_eng = ValuationEngine()
    rec_eng = RecommendationEngine()
    
    results = []
    for idx, row in df_input.iterrows():
        pred_price = prices[idx]
        if pd.isna(pred_price):
            results.append({
                "Predicted Price (₹)": np.nan,
                "Risk Score": np.nan,
                "Risk Category": "Unknown",
                "Recommendation": "N/A",
                "ROI %": np.nan
            })
            continue
            
        city = row.get("city", "Mumbai")
        listing_price = row.get("price", pred_price)
        if pd.isna(listing_price):
            listing_price = pred_price
            
        crime = row.get("crime_index", 45.0)
        age = row.get("property_age", 5)
        infra = row.get("infrastructure_growth_score", 60.0)
        
        roi_data = roi_eng.calculate(listing_price, city)
        risk_data = risk_eng.calculate(crime, age, city, infra)
        val_data = val_eng.classify(listing_price, pred_price)
        rec = rec_eng.recommend(
            roi_data["primary_roi_pct"],
            risk_data["risk_score"],
            val_data["status"],
            risk_data["risk_category"]
        )
        
        results.append({
            "Predicted Price (₹)": pred_price,
            "Risk Score": risk_data["risk_score"],
            "Risk Category": risk_data["risk_category"],
            "Recommendation": rec["recommendation"],
            "ROI %": roi_data["primary_roi_pct"]
        })
    return pd.DataFrame(results)

def render(model, scaler, encoders, feature_cols):
    st.markdown("## 🏷️ Property Price Prediction & Bulk Analyzer")
    
    tab1, tab2 = st.tabs(["🎯 Single Property Predictor", "📥 Bulk Analyzer (CSV Upload)"])
    
    with tab1:
        st.info("Fill in property details to get an AI-predicted market price.")
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("🏙️ City", CITIES, key="pred_city")
            locality = st.selectbox("📍 Locality", LOCALITIES[city], key="pred_loc")
            prop_type = st.selectbox("🏘️ Property Type", PROP_TYPES, key="pred_pt")
            furnished = st.selectbox("🛋️ Furnished Status", FURNISHED, key="pred_furn")
        with col2:
            area = st.number_input("📐 Area (sqft)", 300, 10000, 1200, step=50, key="pred_area")
            bhk = st.selectbox("🛏️ BHK", [1,2,3,4,5], index=1, key="pred_bhk")
            bathrooms = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1, key="pred_bath")
            balcony = st.selectbox("🌇 Balconies", [0,1,2,3], index=1, key="pred_bal")
        with col3:
            floor = st.number_input("🏢 Floor Number", 0, 50, 3, key="pred_floor")
            total_floors = st.number_input("🏗️ Total Floors", 1, 60, 10, key="pred_tfloor")
            age = st.number_input("📅 Property Age (years)", 0, 50, 5, key="pred_age")
            parking = st.selectbox("🚗 Parking", ["Yes","No"], key="pred_park")

        st.markdown("#### 📍 Nearby Facilities & Socioeconomic")
        c1, c2, c3, c4, c5 = st.columns(5)
        hosp   = c1.number_input("🏥 Hospital (km)", 0.1, 30.0, 2.0, key="pred_hosp")
        school = c2.number_input("🏫 School (km)",   0.1, 20.0, 1.5, key="pred_sch")
        metro  = c3.number_input("🚇 Metro (km)",    0.1, 40.0, 3.0, key="pred_metro")
        mall   = c4.number_input("🛍️ Mall (km)",     0.1, 30.0, 4.0, key="pred_mall")
        airport= c5.number_input("✈️ Airport (km)",  1.0, 80.0, 20.0,key="pred_air")
        c6, c7, c8 = st.columns(3)
        pop    = c6.number_input("👥 Pop Density", 1000, 50000, 8000, step=500, key="pred_pop")
        crime  = c7.slider("🚔 Crime Index", 0, 100, 45, key="pred_crime")
        infra  = c8.slider("🏗️ Infra Score", 0, 100, 60, key="pred_infra")

        if st.button("🔮 Predict Price", type="primary", use_container_width=True, key="pred_btn"):
            if model is None:
                st.error("Models not loaded. Run `python pipeline.py` first."); return
            row = _build_input_row(city, locality, area, prop_type, bhk, bathrooms, balcony,
                                   floor, total_floors, age, furnished, parking,
                                   hosp, school, metro, mall, airport, pop, crime, infra)
            pred = _predict(row, model, scaler, encoders, feature_cols)
            pred_l = pred / 1e5
            ppsf   = pred / area

            st.markdown("---")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💰 Predicted Price", f"₹{pred_l:.2f}L", f"₹{pred/1e7:.2f} Cr")
            mc2.metric("📐 Price per sqft", f"₹{ppsf:,.0f}")
            mc3.metric("🏙️ Market", city)

            st.success(f"**AI Price Prediction:** ₹{pred_l:.2f} Lakhs  |  ₹{ppsf:,.0f}/sqft")
            st.session_state["last_prediction"] = {
                "price": pred, "city": city, "area": area, "bhk": bhk,
                "crime": crime, "infra": infra, "age": age, "locality": locality
            }
            
    with tab2:
        st.markdown("### 📊 Bulk Real-Time Property Analyzer")
        st.info("Upload a CSV file containing property details to generate batch predictions, risk scores, and investment recommendations.")
        
        # Download template
        template_data = {
            "city": ["Mumbai", "Bangalore"],
            "locality": ["Bandra", "Whitefield"],
            "area_sqft": [1200, 1800],
            "bhk": [2, 3],
            "property_age": [5, 2],
            "price": [12000000, 15000000],
            "crime_index": [35, 40],
            "infrastructure_growth_score": [75, 80]
        }
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="real_estate_bulk_template.csv",
            mime="text/csv",
            key="bulk_download_template"
        )
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="bulk_csv_uploader")
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df_upload)} rows from CSV.")
                if st.button("🚀 Analyze Bulk Properties", type="primary", use_container_width=True, key="bulk_analyse_btn"):
                    with st.spinner("Analyzing property portfolio..."):
                        res_df = analyze_bulk_rows(df_upload, model, scaler, encoders, feature_cols)
                        final_df = pd.concat([df_upload, res_df], axis=1)
                        
                        st.success("Batch analysis complete!")
                        st.markdown("### 📋 Results Summary")
                        st.dataframe(final_df, use_container_width=True)
                        
                        csv_output = final_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Analyzed Results CSV",
                            data=csv_output,
                            file_name="real_estate_analyzed_results.csv",
                            mime="text/csv",
                            key="bulk_download_results"
                        )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
