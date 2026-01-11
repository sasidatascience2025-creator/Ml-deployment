from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))

@app.route('/')
def home():
    return jsonify({
        "status": "API is running",
        "message": "Use /predict endpoint"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run()
