import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_DIR = r'C:\Users\harip\Desktop\RideWise-Predicting-Bike-sharing-Demand\Modeling\saved_models'
DATASET_PATH = r'C:\Users\harip\Desktop\RideWise-Predicting-Bike-sharing-Demand\Data\EDA_DT_Models.csv'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'DR_scaler.pkl')

# Load best model and scaler
model = joblib.load(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("Bike Sharing Demand Prediction App")
df = pd.read_csv(DATASET_PATH)
st.write("Sample of preprocessed data:")
st.dataframe(df.head())

st.subheader("Enter feature values below :")
st.markdown("Use dropdowns or type manually in the boxes.")
columns = st.columns(4)

# ---- Feature Inputs UI ----
with columns[0]:
    season = st.selectbox("Season", options=[1,2,3,4], format_func=lambda x: {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}[x])
    yr_text = st.text_input("Year (0=2011, 1=2012)", key="yr_text")
    mnth_text = st.text_input("Month (1-12)", key="mnth_text")
with columns[1]:
    holiday = st.selectbox("Holiday", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
    weekday = st.selectbox(
        "Weekday", options=list(range(7)),
        format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x]
    )
    workingday = st.slider("Workingday", min_value=0, max_value=1, step=1, format="%d")
with columns[2]:
    weathersit = st.selectbox("Weathersit", options=[1,2,3,4], format_func=lambda x: {1:"Clear",2:"Mist",3:"Light Rain/Snow",4:"Heavy Rain/Snow"}[x])
    temp_text = st.text_input("Temperature (celsius)", key="temp_text")
    hum_text = st.text_input("Humidity (0-1)", key="hum_text")
with columns[3]:
    windspeed_text = st.text_input("Windspeed (0-1)", key="windspeed_text")
    weekend = st.selectbox("Weekend", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

def pick_input(dropdown_val, textbox_val, dtype=float):
    try:
        val = str(textbox_val).strip()
        if val == "":
            return dropdown_val
        else:
            return dtype(val)
    except Exception:
        return dropdown_val

input_dict = {
    'season': pick_input(season, season, int),
    'yr': pick_input(0, yr_text, int),
    'mnth': pick_input(1, mnth_text, int),
    'holiday': pick_input(holiday, holiday, int),
    'weekday': pick_input(0, weekday, int),
    'workingday': pick_input(workingday, workingday, int),
    'weathersit': pick_input(weathersit, weathersit, int),
    'temp': pick_input(0.5, temp_text, float),
    'hum': pick_input(0.5, hum_text, float),
    'windspeed': pick_input(0.1, windspeed_text, float),
    'weekend': pick_input(weekend, weekend, int)
}

features = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','hum','windspeed','weekend']
input_df = pd.DataFrame([input_dict])

try:
    X_input = scaler.transform(input_df[features])
    if st.button("Predict Demand (cnt)"):
        cnt_pred = model.predict(X_input)
        cnt_pred_final = int(np.round(cnt_pred[0]**2, 0))  # Inverse sqrt if needed (else use cnt_pred[0])
        st.success(f"Predicted bike demand (cnt): {cnt_pred_final}")

    # --- Predicted vs Actual Graph for test dataset ---
    st.subheader("Predicted vs Actual: Test Dataset")
    df_pred = df.copy()
    X_test_all = scaler.transform(df_pred[features])
    y_test_actual = df_pred["cnt"].values if "cnt" in df_pred.columns else np.zeros(df_pred.shape[0])
    y_test_pred = model.predict(X_test_all)
    y_test_pred_final = np.round(y_test_pred**2, 0)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test_actual, y_test_pred_final, color='royalblue', alpha=0.6, label='Predictions')
    ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--', label='Perfect (y=x)')
    ax.set_xlabel("Actual Count")
    ax.set_ylabel("Predicted Count")
    ax.set_title("Actual vs Predicted Bike Demand (Test Data)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Input Error: {e}")
