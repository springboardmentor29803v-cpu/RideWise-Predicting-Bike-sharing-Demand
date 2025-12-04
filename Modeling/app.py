# =========================================
# RideWise Bike Sharing Prediction - PRODUCTION VERSION
# =========================================

import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------
#                 PATH CONFIGURATION 
# ---------------------------------------------------------
BASE_DIR = r"C:\Users\harip\Desktop\RideWise-Predicting-Bike-sharing-Demand"
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Modeling", "saved_models", "bestModel")

DATASET_PATH = os.path.join(DATA_DIR, "preprocessed_day.csv")
NUM_COLS = ["temp", "atemp", "hum", "windspeed"]  # Scaled features


# ---------------------------------------------------------
#                      UTIL FUNCTIONS
# ---------------------------------------------------------
def load_dataset(path):
    """Load dataset safely with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def load_model(path):
    """Load ML model with pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scaler(path):
    """Load scaler if exists."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------
#              PREPROCESS USER INPUT
# ---------------------------------------------------------
def preprocess_user_input(raw_input, df_training, scaler=None):
    """
    Convert real-world user input into model-ready format:
    - One-hot encode categorical values
    - Add missing features
    - Apply scaling to numeric fields
    """

    X = pd.DataFrame([raw_input])

    # ---------- ONE-HOT ENCODING ----------
    categories = {
        "season": [1, 2, 3, 4],
        "mnth": list(range(1, 13)),
        "weekday": list(range(0, 7)),
        "weathersit": [1, 2, 3],
    }

    for col, values in categories.items():
        for v in values:
            X[f"{col}_{v}"] = 1 if X[col].iloc[0] == v else 0

    # Drop original columns
    X.drop(["season", "mnth", "weekday", "weathersit"], axis=1, inplace=True)

    # ---------- ADD MISSING COLUMNS ----------
    for col in df_training.columns:
        if col != "cnt" and col not in X.columns:
            X[col] = 0

    # ---------- SCALE NUMERIC FIELDS ----------
    if scaler:
        X[NUM_COLS] = scaler.transform(X[NUM_COLS])

    # Final ordering
    final_cols = [c for c in df_training.columns if c != "cnt"]
    X = X[final_cols]

    return X


# =========================================================
#                     STREAMLIT UI
# =========================================================
st.title("🚴 RideWise – Bike Sharing Demand Prediction")
st.markdown("### Predict daily rider demand using ML models trained on historical data.")


# ---------------------------------------------------------
#               LOAD TRAINING DATASET
# ---------------------------------------------------------
try:
    df = load_dataset(DATASET_PATH)
    st.success(" Dataset loaded successfully!")
except Exception as e:
    st.error(str(e))
    st.stop()

if st.checkbox("Show Training Dataset"):
    st.write(df.head())


# ---------------------------------------------------------
#                 LOAD MODEL & SCALER
# ---------------------------------------------------------
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

if not model_files:
    st.error(" No model files found in the directory.")
    st.stop()

selected_model_file = st.selectbox("Select Model to Use", model_files)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, selected_model_file)

try:
    model = load_model(BEST_MODEL_PATH)
    st.success(f" Model Loaded: **{selected_model_file}**")
except Exception as e:
    st.error(str(e))
    st.stop()

# Load scaler
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
scaler = load_scaler(SCALER_PATH)

if scaler:
    st.info("🔧 Scaler loaded – Numeric fields will be transformed.")


# ---------------------------------------------------------
#                   USER REAL WORLD INPUTS
# ---------------------------------------------------------
st.subheader(" Enter Real-World Input Features")

input_features = {}

# -------------- NUMERIC INPUTS (2 per row) --------------
st.markdown("### 🌡 Weather Inputs")

numeric_fields = ["temp", "atemp", "hum", "windspeed"]
numeric_defaults = {"temp": 0.3, "atemp": 0.3, "hum": 0.6, "windspeed": 0.2}

for i in range(0, len(numeric_fields), 2):
    cols = st.columns(2)
    for j, field in enumerate(numeric_fields[i : i + 2]):
        input_features[field] = cols[j].number_input(
            f"{field.upper()}",
            value=float(numeric_defaults[field]),
            help="These values are normalized between 0 and 1.",
        )

# -------------- CATEGORICAL INPUTS (2 per row) ----------
st.markdown("###  Categorical Inputs")

cat_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
cat_defaults = {
    "season": 1, "yr": 1, "mnth": 6,
    "holiday": 0, "weekday": 3,
    "workingday": 1, "weathersit": 1
}

cat_options = {
    "season": [1, 2, 3, 4],
    "yr": [0, 1],
    "mnth": list(range(1, 13)),
    "holiday": [0, 1],
    "weekday": list(range(0, 7)),
    "workingday": [0, 1],
    "weathersit": [1, 2, 3],
}

for i in range(0, len(cat_features), 2):
    cols = st.columns(2)
    for j, field in enumerate(cat_features[i : i + 2]):
        input_features[field] = cols[j].selectbox(
            f"{field.upper()}",
            options=cat_options[field],
            index=cat_options[field].index(cat_defaults[field]),
        )


# ---------------------------------------------------------
#                      PREDICTION
# ---------------------------------------------------------
if st.button("🚀 Predict Demand"):
    try:
        X_input = preprocess_user_input(input_features, df, scaler)
        prediction = int(model.predict(X_input)[0])
        st.success(f"🔮 **Predicted Bike Demand: {prediction} riders**")

    except Exception as e:
        st.error(f"❌ Prediction Failed: {str(e)}")


# ---------------------------------------------------------
#        OPTIONAL PLOT – TRAINING DATA DISTRIBUTION
# ---------------------------------------------------------
if st.checkbox("Show Demand Distribution Plot"):
    st.write("### 📊 Training Dataset Demand Distribution")
    plt.figure(figsize=(10, 5))
    df["cnt"].hist(bins=30)
    plt.xlabel("Bike Count")
    plt.ylabel("Frequency")
    st.pyplot(plt)
