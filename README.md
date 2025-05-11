#  AmISafe Crime Prediction API

AmiSafe is a Flask-based REST API that predicts the **top 3 most likely crime types** based on location (latitude, longitude) and time (month-year). The API leverages a pre-trained machine learning model and integrates with real UK Police crime data to improve prediction accuracy.

---

## 🚀 Features

* Predicts **top 3 crime types** given latitude, longitude, and month.
* Fetches **real-time contextual crime data** from a UK Police API.
* Uses advanced **feature encoding** with location and outcome metadata.
* Outputs prediction **probabilities** and **contextual details**.
* Built with **Flask**, **NumPy**, **Pickle**, and **Scikit-learn**.

---

## 🧠 Model Inputs

The prediction model uses the following features:

* `year`: Extracted from the month input.
* `month_sin` & `month_cos`: Cyclic encoding of the month.
* `coord_group_encoded`: Rounded and encoded lat/lon coordinates.
* `outcome_encoded`: Placeholder fallback encoding.
* `status_encoded`: Most common crime outcome status in the area.
* `location_encoded`: Most common location type in the area.
* `lsoa_encoded`: Most likely street/LSOA group name in the area.

---

## 📦 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/amisafe-api.git
cd amisafe-api
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

### 3. Add Model & Encoder Files

Place the following pre-trained files in the root directory:

* `crime_model.pkl`
* `crime_encoder.pkl`
* `coord_encoder.pkl`
* `Last_outcome_category_encoder.pkl`
* `LSOA_group_encoder.pkl`
* `Outcome_Status_encoder.pkl`
* `Location_type_encoder.pkl`

> These files must match the ones used during model training.

---

## 🧪 Running the API

```bash
python app.py
```

> The API will be available at: `http://localhost:5000`

---

## 🔁 API Endpoints

### `GET /`

Returns a simple status message.

```bash
curl http://localhost:5000/
```

### `POST /predict`

Predict the top 3 crime types based on location and time.

#### Request Body (JSON)

```json
{
  "latitude": 52.4862,
  "longitude": -1.8904,
  "month": "2024-07"
}
```

#### Response (JSON)

```json
{
  "top_3_predicted_crime_types": ["burglary", "vehicle crime", "anti-social behaviour"],
  "probabilities": {
    "burglary": 0.34,
    "vehicle crime": 0.29,
    "anti-social behaviour": 0.15
  },
  "location_features": {
    "location_type": "On or near Road",
    "outcome_status": "Under investigation",
    "street_name": "Some Street"
  },
  "coordinates": {
    "latitude": 52.49,
    "longitude": -1.89
  },
  "model_version": "1.0"
}
```

---

## 🌐 External API Integration

This app fetches historical crime context from:

```
https://pok7e31yel.execute-api.eu-west-2.amazonaws.com/prod/api/v1/crimes/nearby
```

> Requires internet access for successful contextual enrichment.

---

## 🛠️ Tech Stack

* **Backend**: Flask
* **ML Model**: Scikit-learn (XGBoost or other)
* **Encoding**: LabelEncoder (Pickle)
* **Hosting Ready**: Deployable on any cloud (e.g., AWS, Heroku)

---

## 📁 Folder Structure

```
amisafe-api/
│
├── app.py                     # Main Flask application
├── crime_model.pkl            # Trained ML model
├── *_encoder.pkl              # Encoders for categorical features
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧹 Future Work

* Add authentication & rate-limiting
* Add confidence score explanations
* Support broader API radius
* Provide location heatmaps (optional front-end)

---

## 🔒 Disclaimer

This prediction API is for **educational and research purposes only**. Do not use it for real-world law enforcement or critical decision-making without proper validation.

---

## 🤝 License

MIT License © Dilshan Udara Kodithuwakku Maddege
