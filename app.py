from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "service": "ML Prediction API",
        "endpoint": "/predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features")

    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
