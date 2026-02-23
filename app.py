import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "wine_lr.joblib")

_loaded = joblib.load(MODEL_PATH)
if isinstance(_loaded, dict) and "model" in _loaded:
    model = _loaded["model"]
else:
    model = _loaded

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "missing_fields", "missing": missing}), 400

    x = np.array([[float(data[f]) for f in FEATURES]], dtype=float)
    pred = float(model.predict(x)[0])
    return jsonify({"prediction": pred})

@app.get("/")
def index():
    return (
        "Wine Quality Predictor is running ✅\n"
        "Use POST /predict with JSON to get a prediction.\n"
    )

