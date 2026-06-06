from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        job = request.form['job']
        experience = int(request.form['experience'])

        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job,
            "Years of Experience": experience
        }])

        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Predicted Salary: ${prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)