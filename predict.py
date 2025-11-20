import numpy as np
import joblib
import json

# ===============================
# Load model, scaler, and feature names
# ===============================

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)


# ===============================
# Prediction Function
# ===============================

def predict_stock(features_dict):
    """
    Predict using the trained ML model.
    Input: dict containing feature values
    Output: float (probability)
    """

    # Convert dict → vector (same order as during training)
    try:
        data = np.array([[features_dict[feat] for feat in feature_names]])
    except KeyError as e:
        raise ValueError(f"Missing feature in request: {str(e)}")

    # Scale input using trained scaler
    data_scaled = scaler.transform(data)

    # Predict probability (classification model)
    proba = model.predict_proba(data_scaled)[0][1]

    return float(proba)  # return simple float for FastAPI
