from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the expected feature columns based on the original training data
# This is crucial for ensuring the input data has the correct format and order
# Based on the previous steps, X.columns had: 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request body
    data = request.get_json(force=True)

    # Convert the input JSON data into a pandas DataFrame
    # Ensure the columns are in the same order as expected by the model
    try:
        input_df = pd.DataFrame(data, columns=feature_columns)
    except ValueError as e:
        return jsonify({"error": f"Invalid input data format: {e}. Ensure all required columns are present and data is correctly structured."}), 400

    # Preprocess the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Make predictions using the loaded model
    predictions = model.predict(scaled_input)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

# To run the Flask application in a Colab environment, it's common to use run_simple from werkzeug.serving
# However, for demonstration, we will include the standard app.run() but will explain how to run it externally.
# If you were running this as a standalone script, you would use:
if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)

print("Flask application configured. To run it, save this code as a .py file (e.g., app.py) and execute 'python app.py'.")
print("Then, you can send POST requests to http://127.0.0.1:5000/predict with JSON data for prediction.")
