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

/* ------------------ GLOBAL SPACING ------------------ */
.container-box {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 35px;
    margin: 25px auto;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.4);
    width: 80%;
}

/* Section titles */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #212121;
    margin-bottom: 18px;
    border-left: 6px solid #30CFD0;
    padding-left: 12px;
}

/* Clean input grid */
.input-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 28px;
}

/* Makes widgets uniform */
.stSelectbox, .stNumberInput, .stSlider {
    padding: 3px;
}

.stSelectbox > div > div, .stNumberInput input {
    border-radius: 10px !important;
}

.stButton>button {
    margin-top: 15px;
}

/* Prediction result highlight */
.result-box {
    margin-top: 18px;
    padding: 25px;
    border-radius: 12px;
    background: #ffffff;
    border-left: 6px solid #330867;
    box-shadow: 0 3px 12px rgba(0,0,0,0.1);
}

/* Center predict button */
.predict-box {
    text-align: center;
    margin-top: 30px;
}

/* Responsive support */
@media(max-width:900px){
    .input-grid { grid-template-columns: repeat(2,1fr); }
}
@media(max-width:600px){
    .input-grid { grid-template-columns: repeat(1,1fr); }
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# LOAD RESOURCES
# -------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(r"/Users/harshita/vs_code/Infosys_springboard/RideWise-Predicting-Bike-sharing-Demand/Saved_model_files/gradient_boost.pkl")
        feature_names = joblib.load(r"/Users/harshita/vs_code/Infosys_springboard/RideWise-Predicting-Bike-sharing-Demand/feature_names.pkl")
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Error loading files: {e}")
        st.stop()

model, feature_names = load_resources()

# -------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------
# Layout: Use a narrow center column to keep the logo tight and centered
c1, c2, c3 = st.columns([3, 4, 3])

#with c2:
    # A vibrant, modern 3D-style cyclist illustration
    # st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=160)

# Centered Headings with enhanced typography
st.markdown('<h1 class="title-text"> RideWise</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Urban Bike Mobility Forecaster</p>', unsafe_allow_html=True)

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
if st.button("PREDICT DEMAND"):
    
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