import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="RideWise | Bike Prediction",
    page_icon="🚴",
    layout="centered"
)

# -------------------------------------------------------
# CUSTOM CSS & COLOR THEME
# -------------------------------------------------------
st.markdown("""
    <style>
    /* IMPORT FONT */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* BACKGROUND GRADIENT */
    .stApp {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* MAIN TITLE STYLING */
    .title-text {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #30CFD0 0%, #330867 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .subtitle-text {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* CARD CONTAINERS */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 25px;
    }

    /* SECTION HEADERS */
    .section-header {
        color: #330867;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 20px;
        border-left: 5px solid #30CFD0;
        padding-left: 15px;
    }

    /* BUTTON STYLING */
    .stButton>button {
        background: linear-gradient(90deg, #30CFD0 0%, #330867 100%);
        color: white;
        border: none;
        padding: 18px 40px;
        font-size: 20px;
        font-weight: 600;
        border-radius: 50px;
        width: 100%;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 5px 15px rgba(51, 8, 103, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(51, 8, 103, 0.4);
    }

    /* RESULT BOX */
    .stSuccess {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        border: none;
        border-radius: 15px;
        padding: 15px;
        font-size: 1.1rem;
    }

    /* INFO BOX */
    .stInfo {
        background-color: #f8f9fa;
        color: #2c3e50;
        border-left: 4px solid #330867;
    }
    
    /* INPUT WIDGETS */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #444;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD RESOURCES
# -------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\Saved_Models_files\gradient_boost(CV).pkl")
        feature_names = joblib.load(r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\Saved_Models_files\feature_names.pkl")
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Error loading files: {e}")
        st.stop()

model, feature_names = load_resources()

# -------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------
# Layout: Use a narrow center column to keep the logo tight and centered
c1, c2, c3 = st.columns([3, 2, 3])

with c2:
    # A vibrant, modern 3D-style cyclist illustration
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=160)

# Centered Headings with enhanced typography
st.markdown('<h1 class="title-text"> RideWise</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">🚴‍♂️Urban Bike Mobility Forecaster</p>', unsafe_allow_html=True)

# -------------------------------------------------------
# INPUT CARDS
# -------------------------------------------------------

# --- CARD 1: TIME SETTINGS ---
with st.container():
    st.markdown('<div class="section-header">📅 Date & Schedule</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        yr = st.selectbox("Year", [2011, 2012], help="Historical data")
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    
    with c2:
        mnth = st.number_input("Month (1-12)", 1, 12, 7)
        weekday = st.selectbox("Day", options=[0,1,2,3,4,5,6], 
                            format_func=lambda x: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][x])
    
    with c3:
        workingday = st.selectbox("Is it a Working Day?", ["Yes", "No"])
        holiday = st.selectbox("Is it a Holiday?", ["Yes", "No"], index=1)

# --- CARD 2: WEATHER SETTINGS ---
with st.container():
    st.markdown('<div class="section-header">🌤️ Weather Conditions</div>', unsafe_allow_html=True)
    
    weathersit = st.selectbox("General Outlook", ["Clear", "Mist", "Snow/Rain"])
    st.write("") 

    # Interactive Sliders
    temp_input = st.slider("🌡️ Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
    hum_input = st.slider("💧 Humidity (%)", 0, 100, 50, 1)
    windspeed_input = st.slider("💨 Windspeed (m/s)", 0.0, 50.0, 12.0, 0.1)

# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------

st.write("")
if st.button("✨ PREDICT DEMAND"):
    
    # 1. Processing Logic
    season_map = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
    season_val = season_map[season]
    
    yr_val = 1 if yr == 2012 else 0
    working_val = 1 if workingday == "Yes" else 0
    holiday_val = 1 if holiday == "Yes" else 0
    weekend_val = 1 if weekday in [0, 6] else 0

    # Scale values
    temp = temp_input / 41
    hum = hum_input / 100
    windspeed = windspeed_input / 67

    input_data = {
        "yr": yr_val,
        "holiday": holiday_val,
        "workingday": working_val,
        "weekend": weekend_val,
        "temp": temp,
        "hum": hum,
        "windspeed": windspeed,
        "season_2": 1 if season_val == 2 else 0,
        "season_3": 1 if season_val == 3 else 0,
        "season_4": 1 if season_val == 4 else 0,
    }

    # One-hot encoding loops
    for i in range(2, 13):
        input_data[f"mnth_{i}"] = 1 if mnth == i else 0
    for i in range(1, 7):
        input_data[f"weekday_{i}"] = 1 if weekday == i else 0

    input_data["weathersit_2"] = 1 if weathersit == "Mist" else 0
    input_data["weathersit_3"] = 1 if weathersit == "Snow/Rain" else 0

    # Create DF & Align Columns
    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # 2. Prediction
    prediction = model.predict(input_df)[0]
    final_pred = int(prediction)
    
    # 3. Display Results
    st.write("---")
    st.success(f"### 🚴 Predicted Rentals: **{final_pred}**")

    # 4. Generate Report
    reason_html = ""
    
    if temp_input < 5:
        reason_html += "❄️ <b>Cold:</b> Low temps are reducing demand.<br>"
    elif temp_input > 30:
        reason_html += "🔥 <b>Heat:</b> High temps are reducing rider comfort.<br>"
    else:
        reason_html += "🌤️ <b>Pleasant:</b> Ideal temperature for cycling.<br>"

    if hum_input > 80:
        reason_html += "💧 <b>Humid:</b> High humidity is discouraging riders.<br>"
    
    if weathersit == "Snow/Rain":
        reason_html += "🌧️ <b>Weather:</b> Rain/Snow is a major deterrent.<br>"
    elif weathersit == "Mist":
        reason_html += "🌫️ <b>Visibility:</b> Mist has a moderate negative impact.<br>"
    else:
        reason_html += "☀️ <b>Clear:</b> Clear skies are boosting demand.<br>"

    # UPDATED INSIGHT REPORT with DARK FONTS
    st.markdown(f"""
    <div style="background-color: white; padding: 25px; border-radius: 12px; border-left: 6px solid #330867; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="color: #000000; margin-top:0; font-weight: 700;">📊 Insight Report</h3>
        <div style="color: #222222; font-size: 16px; line-height: 1.8; font-weight: 500;">
            {reason_html}
        </div>
    </div>
    """, unsafe_allow_html=True)