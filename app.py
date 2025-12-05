import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
MODEL_FOLDER = r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\Saved_Models_files"
DATA_FOLDER = r"D:\RideWise-Predicting-Bike-sharing-Demand\Data"

best_model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
training_columns_path = os.path.join(DATA_FOLDER, "training_columns.pkl")
numeric_columns_path = os.path.join(DATA_FOLDER, "numeric_columns.pkl")
scaler_path = os.path.join(MODEL_FOLDER, "scaler.pkl")

# Load artifacts
model = joblib.load(best_model_path)
training_columns = joblib.load(training_columns_path)
numeric_cols = joblib.load(numeric_columns_path)

# Load scaler if exists
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = None

def needs_scaling(model):
    return isinstance(model, (Lasso, Ridge, ElasticNet))

# -----------------------------------------------------
# PREPROCESS SINGLE INPUT
# -----------------------------------------------------
def preprocess(input_dict):
    df = pd.DataFrame([input_dict])

    # Correct weekend logic
    df["weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # Apply OHE
    cat_cols = ["season", "mnth", "weekday", "weathersit"]
    for col in cat_cols:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Add missing columns
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    df = df[training_columns]

    # Apply scaler only for linear models
    if scaler is not None and needs_scaling(model):
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# -----------------------------------------------------
# STREAMLIT MODERN UI
# -----------------------------------------------------
st.set_page_config(page_title="RideWise Bike Demand Predictor", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>
        🚴 RideWise – Bike Demand Predictor
    </h1>

    <p style='text-align: center; font-size: 18px; color: #444;'>
        Enter your environmental & calendar variables to predict bike rental demand instantly.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# INPUT FORM
# -----------------------------------------------------
st.markdown("### 📌 Enter Prediction Inputs")

col1, col2 = st.columns(2)

with col1:
    yr = st.selectbox("Year (0 = 2011, 1 = 2012)", [0, 1])
    holiday = st.selectbox("Holiday", [0, 1])
    workingday = st.selectbox("Working Day", [0, 1])
    temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.6)

with col2:
    windspeed = st.slider("Wind Speed (normalized)", 0.0, 1.0, 0.25)
    season = st.selectbox("Season", [1, 2, 3, 4])
    mnth = st.selectbox("Month", list(range(1, 13)))
    weekday = st.selectbox("Weekday (0=Sun ... 6=Sat)", list(range(7)))
    weathersit = st.selectbox("Weather Situation", [1, 2, 3])

# Predict button
if st.button("🔮 Predict Bike Count", use_container_width=True):

    input_data = {
        "yr": yr,
        "holiday": holiday,
        "workingday": workingday,
        "temp": temp,
        "hum": hum,
        "windspeed": windspeed,
        "season": season,
        "mnth": mnth,
        "weekday": weekday,
        "weathersit": weathersit
    }

    processed = preprocess(input_data)
    prediction = int(model.predict(processed)[0])

    st.markdown(
        f"""
        <div style="
            background-color:#E8F5E9;
            padding:20px;
            border-radius:15px;
            margin-top:20px;
            text-align:center;
            box-shadow:0 0 10px rgba(0,0,0,0.1);
        ">
            <h2 style="color:#1B5E20;">🚲 Predicted Bike Rental Count</h2>
            <h1 style="color:#2E7D32; font-size:48px;">{prediction}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
