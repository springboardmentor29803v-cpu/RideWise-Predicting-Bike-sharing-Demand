import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------------

data_path = r"D:\RideWise-Predicting-Bike-sharing-Demand\Data\preprocessed_day.csv"
df = pd.read_csv(data_path , encoding= 'unicode_escape')
model = joblib.load(r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\Saved_Models_files\gradient_boost(CV).pkl")
scaler = joblib.load(r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\Saved_Models_files\scaler.pkl")
feature_names = joblib.load(r"D:\RideWise-Predicting-Bike-sharing-Demand\Modeling\feature_names.pkl")


st.set_page_config(page_title="Bike Rental Prediction",
                   page_icon="🚴",
                   layout="centered")

# -------------------------------------------------------
# APP HEADER
# -------------------------------------------------------
st.title("🚴 Bike Rental Prediction App")
st.markdown("### Enter weather and seasonal information to predict bike rentals")

st.write("---")

# -------------------------------------------------------
# USER INPUT SECTION
# -------------------------------------------------------

st.subheader("📌 Input Features")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        ["Spring", "Summer", "Fall", "Winter"],
        help="Select the season of the year"
    )

    yr = st.selectbox(
        "Year",
        [2011, 2012],
        help="Dataset contains only two years"
    )

    mnth = st.number_input(
        "Month (1-12)",
        min_value=1,
        max_value=12,
        help="Enter month number"
    )

    weekday = st.number_input(
        "Weekday (0=Sun ... 6=Sat)",
        min_value=0,
        max_value=6,
        help="Enter day of week"
    )

with col2:
    workingday = st.selectbox(
        "Working Day",
        ["Yes", "No"],
        help="Yes if working day, No otherwise"
    )

    holiday = st.selectbox(
        "Holiday",
        ["Yes", "No"],
        help="Select Yes if it is a holiday"
    )

    weathersit = st.selectbox(
        "Weather Situation",
        ["Clear", "Mist", "Snow/Rain"],
        help="Select today's weather"
    )

    temp_input = st.text_input(
        "Temperature (°C)",
        placeholder= "Enter temperature like 20.5",
        help="Actual Temperature"
    )

hum_input = st.text_input(
    "Humidity (%)",
    placeholder= "Enter humidity like 65",
    help="Humidity values"
)

windspeed_input = st.text_input(
    "Windspeed (m/s)",
    placeholder= " Enter windspeed like 12.3",
    # help="Windspeed values divided by 67"
)
# -------------------------------------------------------
# PROCESS INPUTS
# -------------------------------------------------------

if st.button("Predict Rentals 🚴"):

    # Convert numeric inputs
    try:
        temp_input = float(temp_input)
        hum_input = float(hum_input)
        windspeed_input = float(windspeed_input)
    except:
        st.error("❌ Please enter valid numeric values.")
        st.stop()

    # Convert categorical to numeric
    season_map = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
    season_val = season_map[season]

    yr_val = 1 if yr == 2012 else 0
    working_val = 1 if workingday == "Yes" else 0
    holiday_val = 1 if holiday == "Yes" else 0

    # Scale values
    temp = temp_input / 41
    hum = hum_input / 100
    windspeed = windspeed_input / 67

    # Create weekend flag based on user input (NOT dataframe)
    weekend_val = 1 if weekday in [0, 6] else 0

    # Create input dictionary
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

    # Month dummies
    for i in range(2, 13):
        input_data[f"mnth_{i}"] = 1 if mnth == i else 0

    # Weekday dummies
    for i in range(1, 7):
        input_data[f"weekday_{i}"] = 1 if weekday == i else 0

    # Weathersit dummies
    input_data["weathersit_2"] = 1 if weathersit == "Mist" else 0
    input_data["weathersit_3"] = 1 if weathersit == "Snow/Rain" else 0

    input_df = pd.DataFrame([input_data])

    # Add missing columns
    for col in feature_names:
     if col not in input_df.columns:
         input_df[col] = 0

    # Reorder
    input_df = input_df[feature_names]


    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"### 🚴 Predicted Bike Rentals: **{int(prediction)}**")


    reason = ""

    if temp_input < 5:
        reason += "❄️ Low temperature reduced expected rentals.\n"
    elif temp_input > 30:
        reason += "🔥 High temperature reduced comfort level.\n"
    else:
        reason += "🌤️ Pleasant temperature supports higher rentals.\n"

    if hum_input > 80:
        reason += "💧 High humidity discourages riders.\n"

    if weathersit == "Snow/Rain":
        reason += "🌧️ Bad weather caused lower rentals.\n"
    elif weathersit == "Mist":
        reason += "🌫️ Misty weather moderately impacts rentals.\n"
    else:
        reason += "☀️ Clear weather encourages more riders.\n"

    st.info("### 🔍 Reason Behind Prediction:\n" + reason)

