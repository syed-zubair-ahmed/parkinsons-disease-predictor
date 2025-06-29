from flask import Flask, request, render_template
import numpy as np
import pickle
import sys

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('parkinsons_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    sys.exit(1)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Form route
@app.route('/form')
def form():
    return render_template('form.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        fields = [
            'fo', 'fhi', 'flo', 'jitter_percent', 'jitter_abs', 'rap', 'ppq',
            'ddp', 'shimmer', 'shimmer_db', 'apq3', 'apq5', 'apq', 'dda',
            'nhr', 'hnr', 'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'
        ]
        
        input_data = [float(request.form[field]) for field in fields]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        result_msg = "Parkinson’s Disease Detected" if prediction == 1 else "No Parkinson’s Disease Detected"

        return render_template('result.html', result=result_msg, probability=round(probability, 2))

    except Exception as e:
        print(f"Prediction error: {e}")
        return "An error occurred during prediction. Please check your input values and try again."

if __name__ == '__main__':
    app.run(debug=True)
