from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)    
from flask_cors import CORS
CORS(app, origins=["http://localhost:5173"])              

# Load the trained model
model = joblib.load("house_price_model.pkl")

# List of feature names in correct order
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

@app.route("/")
def home():
    return "Welcome to the House Price Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        features = [data[feature] for feature in feature_names]
        prediction = model.predict([features])
        return jsonify({"predicted_price": float(round(prediction[0], 2))})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
