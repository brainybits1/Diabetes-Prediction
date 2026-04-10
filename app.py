"""
Flask API for Diabetes Prediction Web App
Endpoints:
  GET  /           → serves index.html
  POST /predict    → returns prediction + probability + risk level
  GET  /model-info → returns feature importances + model stats
"""

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "diabetes_model.pkl")
model_data = joblib.load(MODEL_PATH)

pipeline            = model_data["pipeline"]
FEATURE_COLS        = model_data["feature_cols"]

feature_importances = model_data["feature_importances"]
MODEL_ACCURACY    = model_data["accuracy"]
MODEL_AUC         = model_data["auc"]


def get_risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Moderate"
    else:
        return "High"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Extract features in correct order, default 0 if missing
        features = [float(data.get(col, 0)) for col in FEATURE_COLS]
        X = np.array(features).reshape(1, -1)

        prediction  = int(pipeline.predict(X)[0])
        probability = float(pipeline.predict_proba(X)[0][1])
        risk_level  = get_risk_level(probability)

        # Build per-feature contribution hints
        advice = []
        vals = dict(zip(FEATURE_COLS, features))
        if vals.get("Glucose", 0) > 140:
            advice.append("Your glucose level is high — consult a doctor.")
        if vals.get("BMI", 0) > 30:
            advice.append("BMI indicates obesity — consider weight management.")
        if vals.get("Age", 0) > 45:
            advice.append("Age is a risk factor; regular screening is recommended.")
        if vals.get("Insulin", 0) > 200:
            advice.append("Insulin level is elevated.")

        return jsonify({
            "prediction":  prediction,
            "probability": round(probability * 100, 1),
            "risk_level":  risk_level,
            "advice":      advice,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-info", methods=["GET"])
def model_info():
    # Sort importances descending
    sorted_imp = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )
    return jsonify({
        "accuracy":            round(MODEL_ACCURACY * 100, 2),
        "auc":                 round(MODEL_AUC, 4),
        "feature_importances": sorted_imp,
        "features":            FEATURE_COLS,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
