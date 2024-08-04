from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array([features])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return render_template('page1.html', prediction_text=f'Predicted Grade Class: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
