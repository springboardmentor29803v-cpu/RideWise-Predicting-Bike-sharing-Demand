import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model + scaler
model = joblib.load("bike_sharing_model.pkl")
scaler = joblib.load("scaler.pkl")

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

# ------------------------------------------------------
# REMOVE STREAMLIT'S WHITE BLOCK BARS
# ------------------------------------------------------
st.markdown("""
<style>

    /* Remove Streamlit’s auto white containers */
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

    /* Remove divider-like pill bars */
    .css-1y4p8pa, .css-1wrcr25, .css-6qob1r {
        background: transparent !important;
        box-shadow: none !important;
    }

</style>
""", unsafe_allow_html=True)


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

# Year
year_map = {"2018": 0, "2019": 1}
year_name = col2.selectbox("Year", list(year_map.keys()))
yr = year_map[year_name]

# MONTH CHOICEBOX — NEW
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
month_name = col1.selectbox("Month", list(month_map.keys()))
mnth = month_map[month_name]

# WEATHER
weather_map = {
    "Clear / Few Clouds": 1,
    "Mist / Cloudy": 2,
    "Light Snow / Rain": 3,
    "Heavy Rain / Storm": 4
}
weather_name = col2.selectbox("Weather Situation", list(weather_map.keys()))
weathersit = weather_map[weather_name]

col5, col6 = st.columns(2)

holiday = col5.selectbox("Holiday?", ["No", "Yes"])
holiday = 1 if holiday == "Yes" else 0

workingday = col6.selectbox("Working Day?", ["No", "Yes"])
workingday = 1 if workingday == "Yes" else 0

# WEEKDAY CHOICEBOX — NEW
weekday_map = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
}
weekday_name = st.selectbox("Weekday", list(weekday_map.keys()))
weekday = weekday_map[weekday_name]
is_weekend = 1 if weekday in [0, 6] else 0


# ------------------------------------------------------
# USER INPUT FIELDS FOR WEATHER (FLOAT ALLOWED)
# ------------------------------------------------------
st.markdown("### 🌡 Enter Weather Values ")

col7, col8, col9 = st.columns(3)

temp_real = col7.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=21.0, step=0.1)
hum_real = col8.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
windspeed_real = col9.number_input("Windspeed (km/h)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)


st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# INTERNAL NORMALIZATION (MODEL-FRIENDLY)
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
    "year": [yr],
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

if st.button("🔮 Predict Now"):
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    st.success(f"### ✅ Estimated Bike Rentals: **{prediction} bikes**")
    st.info("Model Used: Random Forest (Best Model)")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
