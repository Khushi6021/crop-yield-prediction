from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)   # ✅ IMPORTANT (must be here)

# Load models
rf_model = joblib.load("rf_model.pkl")
lstm_model = load_model("lstm_model.h5", compile=False)

# Home route
@app.route('/')
def home():
    return "Hybrid Model API Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = np.array([[
        float(data['rainfall']),
        float(data['temperature']),
        float(data['nitrogen']),
        float(data['phosphorus']),
        float(data['potassium'])
    ]])

    # Random Forest prediction
    rf_pred = rf_model.predict(features)[0]

    # LSTM prediction
    lstm_input = features.reshape((1, 1, features.shape[1]))
    lstm_pred = lstm_model.predict(lstm_input)[0][0]

    # Hybrid prediction
    hybrid_pred = (rf_pred + lstm_pred) / 2

    return jsonify({
        "random_forest": round(float(rf_pred), 2),
        "lstm": round(float(lstm_pred), 2),
        "hybrid": round(float(hybrid_pred), 2)
    })

# Run server
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))