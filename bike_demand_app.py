import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ------------------------------------------------------
# Try to load a model that does NOT need 'yr' first.
# If not present, fall back to original model (which expects 'yr').
# ------------------------------------------------------
model_filename_no_yr = "bike_sharing_model_no_yr.pkl"
scaler_filename_no_yr = "scaler_no_yr.pkl"
model_filename_with_yr = "bike_sharing_model.pkl"
scaler_filename_with_yr = "scaler.pkl"

model = None
scaler = None
model_needs_yr = True  # default — assume original requires yr

try:
    model = joblib.load(model_filename_no_yr)
    scaler = joblib.load(scaler_filename_no_yr)
    model_needs_yr = False
    st.write("")  # quiet placeholder
except Exception:
    # fallback
    model = joblib.load(model_filename_with_yr)
    scaler = joblib.load(scaler_filename_with_yr)
    model_needs_yr = True

# ------------------------------------------------------
# PAGE CONFIG + CUSTOM CSS
# ------------------------------------------------------
st.set_page_config(page_title="Bike Demand Predictor", page_icon="🚲", layout="wide")

st.markdown("""
    <style>

        body {
            background-color: #F4F6F9;
        }

        .main-container {
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
            padding: 25px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }

        .app-title {
            text-align: center;
            color: #2C3E50;
            font-size: 38px;
            font-weight: bold;
            margin-top: 20px;
        }

        .section-title {
            color: #34495E;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 15px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)

# Remove Streamlit white blocks
st.markdown("""
<style>
    [data-testid="stBlock"] {
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .css-1y4p8pa, .css-1wrcr25, .css-6qob1r {
        background: transparent !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# SYNCHRONIZATION LOGIC FOR HOLIDAY & WORKING DAY
# ------------------------------------------------------

# Initialize session state for working_day if not present
if 'working_day_state' not in st.session_state:
    # Default to 'Yes' since 'Holiday?' defaults to 'No'
    st.session_state['working_day_state'] = 'Yes'

def update_working_day():
    """Updates the 'Working Day?' state based on the new 'Holiday?' selection."""
    
    # Get the value from the Holiday selectbox, which is stored in st.session_state['holiday_input']
    # 'Yes' -> Holiday (Not Working Day)
    if st.session_state['holiday_input'] == 'Yes':
        st.session_state['working_day_state'] = 'No'
    # 'No' -> Not Holiday (Working Day)
    else:
        st.session_state['working_day_state'] = 'Yes'

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.markdown("<div class='app-title'>🚲 Bike Sharing Demand Predictor</div>", unsafe_allow_html=True)
st.write("### Enter ride conditions below to estimate bike rentals.")
st.markdown("<div class='main-container'>", unsafe_allow_html=True)


# ------------------------------------------------------
# INPUT CARD
# ------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📌 Enter Ride Conditions</div>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)

# Season
season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
season_name = col1.selectbox("Season", list(season_map.keys()))
season = season_map[season_name]

# YEAR HIDDEN INTERNALLY
year_map = {"2018": 0, "2019": 1}
current_year_str = str(datetime.now().year)
yr = year_map.get(current_year_str, 1)

# MONTH
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
month_name = col1.selectbox("Month", list(month_map.keys()))
mnth = month_map[month_name]

# WEATHER (Right column)
weather_map = {
    "Clear / Few Clouds": 1,
    "Mist / Cloudy": 2,
    "Light Snow / Rain": 3,
    "Heavy Rain / Storm": 4
}
weather_name = col2.selectbox("Weather Situation", list(weather_map.keys()))
weathersit = weather_map[weather_name]

# WEEKDAY MOVED HERE — RIGHT COLUMN UNDER WEATHER
weekday_map = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
}
weekday_name = col2.selectbox("Weekday", list(weekday_map.keys()))
weekday = weekday_map[weekday_name]
is_weekend = 1 if weekday in [0, 6] else 0

# Holiday + Working Day row
col5, col6 = st.columns(2)

# --- MODIFIED WIDGETS FOR SYNCHRONIZATION ---

# Holiday Selectbox: Uses a key and an on_change callback
holiday_name = col5.selectbox(
    "Holiday?", 
    options=["No", "Yes"], 
    key='holiday_input',               # Key to store its value in session state
    on_change=update_working_day       # Function to call when value changes
)
holiday = 1 if holiday_name == "Yes" else 0

# Working Day Selectbox: Uses the session state key as its default value
workingday_name = col6.selectbox(
    "Working Day?", 
    options=["No", "Yes"],
    # Read the synchronized value from the session state
    index=["No", "Yes"].index(st.session_state['working_day_state'])
)
workingday = 1 if workingday_name == "Yes" else 0

# --- END MODIFIED WIDGETS ---


# ------------------------------------------------------
# WEATHER INPUTS
# ------------------------------------------------------
st.markdown("### 🌡 Enter Weather Values ")

col7, col8, col9 = st.columns(3)

temp_real = col7.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=21.0, step=0.1)
hum_real = col8.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
windspeed_real = col9.number_input("Windspeed (km/h)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------
temp = temp_real / 41
hum = hum_real / 100
windspeed = windspeed_real / 67


# ------------------------------------------------------
# SUMMARY CARD
# ------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🔍 Input Summary</div>", unsafe_allow_html=True)

summary_df = pd.DataFrame({
    "season": [season],
    "month": [mnth],
    "holiday": [holiday],
    "weekday": [weekday],
    "workingday": [workingday],
    "weather": [weathersit],
    "temp (°C)": [temp_real],
    "humidity (%)": [hum_real],
    "windspeed (km/h)": [windspeed_real],
    "is_weekend": [is_weekend]
})

st.dataframe(summary_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# PREDICTION CARD
# ------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🚀 Predict Bike Demand</div>", unsafe_allow_html=True)

if model_needs_yr:
    input_df = pd.DataFrame({
        "season": [season],
        "yr": [yr],
        "mnth": [mnth],
        "holiday": [holiday],
        "weekday": [weekday],
        "workingday": [workingday],
        "weathersit": [weathersit],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed],
        "is_weekend": [is_weekend]
    })
else:
    input_df = pd.DataFrame({
        "season": [season],
        "mnth": [mnth],
        "holiday": [holiday],
        "weekday": [weekday],
        "workingday": [workingday],
        "weathersit": [weathersit],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed],
        "is_weekend": [is_weekend]
    })

if st.button("🔮 Predict Now"):
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    st.success(f"### ✅ Estimated Bike Rentals: **{prediction} bikes**")
    st.info("Model Used: Random Forest (Best Model)")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)