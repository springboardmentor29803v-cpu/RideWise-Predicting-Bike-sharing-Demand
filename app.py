# app.py
# Full Streamlit app — robust model loading + only the specific UI changes you requested.

import base64
import json
import traceback
from pathlib import Path

import joblib
import pickle
import pandas as pd
import streamlit as st

# ----------------- Background helper (unchanged) -----------------
def set_background_image(image_path: str):
    p = Path(image_path)
    if not p.exists():
        return
    try:
        with open(p, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()

        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(255,255,255,0.55);
            backdrop-filter: blur(4px);
            z-index: 0;
        }}
        .stApp .main {{
            background-color: rgba(255,255,255,0.95);
            padding: 1rem;
            border-radius: 10px;
            color: black;
            z-index: 1;
        }}
        .stSelectbox, .stNumberInput, .stSlider, .stButton, .stTextInput, .stTextArea {{
            z-index: 2 !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

set_background_image("background.jpg")

st.title("Bike Rental Count — Random Forest (rf_rg_model.pkl)")
st.caption("This app prefers rf_rg_model.pkl (place it in the app folder).")

# ----------------- Robust model loading (try joblib then pickle) -----------------
CANDIDATE_MODELS = [
    "rf_rg_model.pkl", "rf_rg.pkl", "model.pkl", "grad_tuned_model.pkl",
    "grad_model.pkl", "regressor.pkl", "ridge_reg.pkl", "lasso_reg.pkl",
    "elastic_net_reg.pkl", "D_tree.pkl", "final_model.pkl", "pipeline.pkl"
]

model = None
MODEL_F = None
load_errors = {}

# Try joblib first for each candidate. If joblib fails, try pickle.
for name in CANDIDATE_MODELS:
    p = Path(name)
    if not p.exists():
        continue
    # try joblib
    try:
        candidate = joblib.load(p)
        if hasattr(candidate, "predict"):
            model = candidate
            MODEL_F = p
            st.success(f"Loaded model (joblib) from: {p.name} — type: {type(candidate).__name__}")
            break
        else:
            load_errors[p.name] = "Loaded with joblib but object has no predict()."
    except Exception as e_job:
        # if joblib failed, try pickle
        try:
            with open(p, "rb") as fh:
                candidate = pickle.load(fh)
            if hasattr(candidate, "predict"):
                model = candidate
                MODEL_F = p
                st.success(f"Loaded model (pickle) from: {p.name} — type: {type(candidate).__name__}")
                break
            else:
                load_errors[p.name] = "Loaded with pickle but object has no predict()."
        except Exception as e_pickle:
            # record the joblib and pickle error (truncated)
            load_errors[p.name] = f"joblib: {type(e_job).__name__}:{str(e_job)[:250]} | pickle: {type(e_pickle).__name__}:{str(e_pickle)[:250]}"

if MODEL_F is None:
    st.error("No usable model found in folder. Please place your .pkl model (rf_rg_model.pkl preferred).")
    if load_errors:
        st.write("Load attempts (truncated):")
        for k, v in load_errors.items():
            st.write(f"- {k}: {v}")
    st.stop()

# Attempt to load optional scaler
scaler = None
SCALER_F = Path("scaler.pkl")
if SCALER_F.exists():
    try:
        scaler = joblib.load(SCALER_F)
        st.info("Loaded scaler from scaler.pkl")
    except Exception:
        st.warning("Failed to load scaler.pkl — proceeding without scaler.")

# ----------------- Fixed feature set (Option A) -----------------
FEATURES = [
    "yr", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "hum", "windspeed",
    "Season1", "Season2", "Season3", "Season4"
]

# Do NOT print the feature list block (you asked to remove that UI block).
st.markdown(f"**Using model file:** `{MODEL_F.name}`")
st.write("Enter feature values and click Predict.")

# ----------------- INPUT CONTROLS (only the requested changes) -----------------
# 1) Year: free input (user types any year). We convert to model's yr (0/1) as you requested earlier.
input_year = st.number_input("Year (enter full year, e.g. 2011, 2012, 2020...)", min_value=1900, max_value=9999, value=2011, step=1)
yr = 0 if input_year <= 2011 else 1  # match earlier mapping: 0->2011, 1->2012 and beyond

# 2) Month (dropdown) — unchanged
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mnth = st.selectbox("Month", list(range(1,13)), format_func=lambda v: f"{v} — {months[v-1]}")

# 3) Holiday (dropdown)
holiday = st.selectbox("Holiday", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

# 4) Weekday (dropdown)
weekdays = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
weekday = st.selectbox("Weekday", list(range(7)), format_func=lambda v: f"{v} — {weekdays[v]}")

# 5) Workingday (dropdown)
workingday = st.selectbox("Working Day", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

# 6) Weathersit (dropdown)
weathers = {
    1: "1 — Clear / Few Clouds / Partly cloudy",
    2: "2 — Mist / Cloudy",
    3: "3 — Light Snow / Light Rain",
    4: "4 — Heavy Rain / Snow"
}
weathersit = st.selectbox("Weather Situation", list(weathers.keys()), format_func=lambda v: weathers[v])

# 7) temp, hum, windspeed → replace sliders with number inputs that accept larger values
#    (you asked to allow bigger values — here we allow 0..100 with default 0.5; adjust as needed)
temp = st.number_input("temp (normalized — allowed range 0.0 → 100.0)", min_value=0.0, max_value=100.0, value=0.5, step=0.01, format="%.4f")
hum = st.number_input("hum (normalized — allowed range 0.0 → 100.0)", min_value=0.0, max_value=100.0, value=0.5, step=0.01, format="%.4f")
windspeed = st.number_input("windspeed (normalized — allowed range 0.0 → 100.0)", min_value=0.0, max_value=100.0, value=0.5, step=0.01, format="%.4f")

# 8) Season: single dropdown (UI only shows Season), we convert internally to Season1..4 one-hot
season_choice = st.selectbox("Season", [1,2,3,4], format_func=lambda x: {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}[x])
season_map = {
    "Season1": 1 if season_choice == 1 else 0,
    "Season2": 1 if season_choice == 2 else 0,
    "Season3": 1 if season_choice == 3 else 0,
    "Season4": 1 if season_choice == 4 else 0
}

# ----------------- Build input dict in correct model order -----------------
# FEATURE ORDER MUST MATCH model training order; we use FEATURES list to assemble.
user_input = {
    "yr": yr,
    "mnth": mnth,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed,
    "Season1": season_map["Season1"],
    "Season2": season_map["Season2"],
    "Season3": season_map["Season3"],
    "Season4": season_map["Season4"]
}

# Form UI + predict action (keeps layout similar to your earlier code)
with st.form("predict_form"):
    st.subheader("Feature inputs (preview)")
    # Show values in two-column layout similar to before
    left, right = st.columns(2)
    for i, col in enumerate(FEATURES):
        val = user_input[col]
        container = left if (i % 2 == 0) else right
        with container:
            st.write(f"**{col}**: {val}")

    submitted = st.form_submit_button("Predict")

# ----------------- Prediction (unchanged behavior except robust loading) -----------------
if submitted:
    try:
        input_df = pd.DataFrame([user_input])[FEATURES]  # ensure correct column order
        st.subheader("Input preview (DataFrame)")
        st.table(input_df.T.rename(columns={0: "value"}))

        X = input_df.astype(float).values

        # apply scaler if present
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                st.warning("Scaler transform failed — proceeding without scaling for this prediction.")

        pred = model.predict(X)
        raw = float(pred[0])

        # NOTE: many example models were trained on sqrt(cnt) — if yours was too, invert by squaring.
        # If your model predicts cnt directly, remove the squaring below.
        try:
            pred_cnt = int(round(raw ** 2))
            st.success(f"Model output (sqrt(cnt)): {round(raw, 4)}")
            st.info(f"Predicted total rentals (cnt) [inverted by squaring]: {pred_cnt}")
        except Exception:
            st.success(f"Model raw output: {round(raw, 4)}")

        # save prediction to csv
        out = pd.DataFrame([user_input])
        out["model_raw"] = raw
        if 'pred_cnt' in locals():
            out["pred_cnt"] = pred_cnt
        out_file = Path("streamlit_last_predictions.csv")
        if out_file.exists():
            out.to_csv(out_file, mode="a", header=False, index=False)
        else:
            out.to_csv(out_file, index=False)
        st.write(f"Saved prediction to {out_file.name}")

    except Exception:
        st.error("Prediction failed — see traceback below.")
        st.text(traceback.format_exc())

# ----------------- Helpful note if the original error appears again -----------------
st.markdown("---")
st.caption(
    "If you see `UnpicklingError: invalid load key '\\x0c'` again, that usually means the model file is a "
    "joblib file but you tried to open it with pickle. This script tries joblib first and then pickle. "
    "If your model file still fails, re-save it with `joblib.dump(your_model, 'rf_rg_model.pkl')` or "
    "`pickle.dump(your_model, open('model.pkl','wb'))` depending on how you saved it."
)
