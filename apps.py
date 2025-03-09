from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        house_area = request.form.get('area')
        bedrooms = request.form.get('bedrooms')
        floors = request.form.get('floors')

        # Validate inputs
        if not house_area or not bedrooms or not floors:
            return render_template('predict.html', prediction_text="⚠ Please enter all details.")

        # Convert inputs to appropriate data types
        house_area = float(house_area)
        bedrooms = int(bedrooms)
        floors = int(floors)

        # Ensure positive values
        if house_area <= 0 or bedrooms <= 0 or floors <= 0:
            return render_template('predict.html', prediction_text="⚠ All inputs must be positive numbers.")

        # Prepare input data
        input_data = np.array([[house_area, bedrooms, floors]])

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        return render_template('predict.html', prediction_text=f"🏠 Predicted House Price: ₹{round(predicted_price, 2)} Lakh")

    except ValueError:
        return render_template('predict.html', prediction_text="⚠ Invalid input. Please enter valid numbers.")
    except Exception as e:
        return render_template('predict.html', prediction_text=f"⚠ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)