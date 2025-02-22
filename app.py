from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('music-persist.joblib')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get values from the form
            age = request.form['age']
            gender_str = request.form['gender']

            # Convert gender to numeric (Male -> 0, Female -> 1)
            gender = 0 if gender_str == "Male" else 1

            # Create DataFrame for the input with correct feature names
            input_data = pd.DataFrame([[age, gender]], columns=['age', 'gender'])

            # Predict the genre
            prediction = model.predict(input_data)

            # Return the prediction result
            return render_template('index.html', prediction=prediction[0], age=age, gender=gender_str)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
