from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve data from the form
            features = [float(request.form[feature]) for feature in [
                "Annual_Income", "Monthly_Inhand_Salary",
                "Num_Bank_Accounts", "Num_Credit_Card",
                "Interest_Rate", "Num_of_Loan",
                "Delay_from_due_date", "Num_of_Delayed_Payment",
                "Credit_Mix", "Outstanding_Debt",
                "Credit_History_Age", "Monthly_Balance"
            ]]
            
            # Scale the features
            scaled_features = scaler.transform([features])
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            return render_template('index.html', prediction_text=f'Predicted Credit Score: {prediction}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
