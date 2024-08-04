from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
app.config['DEBUG'] = True


try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly.pkl')  # Load polynomial feature transformer
except Exception as e:
    print(f"Error loading model, scaler, or polynomial features: {e}")



@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        age = float(request.form['age'])
        study_time = float(request.form['studyTime'])
        absences = float(request.form['absences'])
        gpa = float(request.form['gpa'])
        parental_education = float(request.form['parentalEducation'])

        # Print features for debugging
        print(f"Age: {age}, Study Time: {study_time}, Absences: {absences}, GPA: {gpa}, Parental Education: {parental_education}")

        features = np.array([[age, study_time, absences, gpa, parental_education]])
        print("Features array:", features)

        # Ensure the scaler and model are loaded
        if 'scaler' not in globals() or 'model' not in globals() or 'poly' not in globals():
            raise Exception("Scaler, model, or polynomial features are not loaded")

        # Apply polynomial feature transformation
        features_poly = poly.transform(features)
        print("Polynomial features:", features_poly)

        features_scaled = scaler.transform(features_poly)
        print("Scaled features:", features_scaled)

        prediction = model.predict(features_scaled)
        print("Prediction:", prediction)

        return render_template('page1.html', prediction_text=f'Predicted Grade Class: {prediction[0]}')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('page1.html', prediction_text='Error in prediction')



if __name__ == "__main__":
    app.run(debug=True)

