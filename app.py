from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("salary_model.pkl", "rb"))

# Load encoders
gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))
education_encoder = pickle.load(open("education_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html", result_html="")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    gender = request.form["gender"]
    education = request.form["education"]
    job_title = request.form["job_title"]
    experience = float(request.form["experience"])

    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_title])[0]

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Job Title": [job_encoded],
        "Years of Experience": [experience],
    })

    prediction = model.predict(input_data)[0]

    result_html = f"""
    <div class='result'>
        <h2>💰 Predicted Salary</h2>
        <div class='salary'>
            Rs.{prediction:,.2f}
        </div>
    </div>
    """

    return render_template(
        "index.html",
        result_html=result_html
    )


if __name__ == "__main__":
    app.run(debug=True)