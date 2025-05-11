from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and encoders (with error handling)
try:
    model = pickle.load(open("crime_model.pkl", "rb"))
    crime_encoder = pickle.load(open("crime_encoder.pkl", "rb"))
    coord_encoder = pickle.load(open("coord_encoder.pkl", "rb"))
    outcome_encoder = pickle.load(open("Last_outcome_category_encoder.pkl", "rb"))
    lsoa_encoder = pickle.load(open("LSOA_group_encoder.pkl", "rb"))
    status_encoder = pickle.load(open("Outcome_Status_encoder.pkl", "rb"))
    location_encoder = pickle.load(open("Location_type_encoder.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model/encoders: {str(e)}")

CRIME_API = "https://pok7e31yel.execute-api.eu-west-2.amazonaws.com/prod/api/v1/crimes/nearby"


def fetch_historical_features(lat, lon):
    """Fetch realistic features from crime API with proper handling"""
    params = {"latitude": lat, "longitude": lon, "radius": 1}

    try:
        response = requests.get(CRIME_API, params=params, timeout=3)
        if response.status_code == 200:
            crimes = response.json()

            # Handle location_type
            location_types = [c.get("location_type") for c in crimes if c.get("location_type")]
            dominant_location_type = max(set(location_types), key=location_types.count) if location_types else \
            location_encoder.classes_[0]

            # Handle outcome_status["category"]
            outcome_categories = [
                c["outcome_status"]["category"]
                for c in crimes
                if c.get("outcome_status") and c["outcome_status"].get("category")
            ]
            dominant_outcome = max(set(outcome_categories), key=outcome_categories.count) if outcome_categories else \
            status_encoder.classes_[0]

            # Handle street_name
            street_name = (
                crimes[0]["location"]["street"]["name"]
                if crimes and crimes[0].get("location", {}).get("street", {}).get("name")
                else lsoa_encoder.classes_[0]
            )

            return {
                "location_type": dominant_location_type,
                "outcome_status": dominant_outcome,
                "street_name": street_name
            }

    except requests.exceptions.RequestException:
        return None


@app.route('/')
def index():
    return "Crime Prediction API is running."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'month' not in data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({"error": "Missing month, latitude or longitude"}), 400

        try:
            # Process time
            month = datetime.strptime(data['month'], "%Y-%m").month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            year = int(data['month'].split('-')[0])

            # Coordinates → coord_group
            lat = round(float(data['latitude']), 2)
            lon = round(float(data['longitude']), 2)
            coord_group = f"{lat}_{lon}"
            coord_group_encoded = coord_encoder.transform([coord_group])[0]

            # Get historical features
            historical = fetch_historical_features(lat, lon)

            if not historical:
                return jsonify({"error": "Could not fetch location history"}), 400

            # Fallback-safe encodings
            outcome_encoded = outcome_encoder.transform(["Status update unavailable"])[0]

            outcome_status = historical["outcome_status"]
            if outcome_status not in status_encoder.classes_:
                outcome_status = status_encoder.classes_[0]
            status_encoded = status_encoder.transform([outcome_status])[0]

            location_type = historical["location_type"]
            if location_type not in location_encoder.classes_:
                location_type = location_encoder.classes_[0]
            location_encoded = location_encoder.transform([location_type])[0]

            street_name = historical["street_name"]
            if street_name not in lsoa_encoder.classes_:
                street_name = lsoa_encoder.classes_[0]
            lsoa_encoded = lsoa_encoder.transform([street_name])[0]

            print(year,
                  month_sin,
                  month_cos,
                  coord_group_encoded,
                  outcome_encoded,
                  status_encoded,
                  location_encoded,
                  lsoa_encoded)

            # Final feature vector
            features = np.array([[
                year,
                month_sin,
                month_cos,
                coord_group_encoded,
                outcome_encoded,
                status_encoded,
                location_encoded,
                lsoa_encoded
            ]])

            # Predict probabilities
            proba = model.predict_proba(features)[0]
            print(proba)
            # Get top 3 predicted crime types
            top3_indices = np.argsort(proba)[::-1][:3]
            top3_crime_types = crime_encoder.inverse_transform(top3_indices)

            return jsonify({
                "top_3_predicted_crime_types": list(top3_crime_types),
                "probabilities": {crime: float(proba[i]) for i, crime in zip(top3_indices, top3_crime_types)},
                "location_features": historical,
                "coordinates": {"latitude": lat, "longitude": lon},
                "model_version": "1.0"
            })

        except ValueError as e:
            return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
