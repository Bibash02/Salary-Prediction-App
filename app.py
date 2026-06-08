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

    # 1. Get input from form
    age = int(request.form["age"])
    experience = float(request.form["experience"])
    gender = request.form["gender"]
    education = request.form["education"]
    job_title = request.form["job_title"]

    # 2. Encode categorical data
    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_title])[0]

    # 3. Create input DataFrame
    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        education_encoded,
        job_encoded,
        experience
    ]], columns=[
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience"
    ])

    # 4. Prediction
    prediction = model.predict(input_data)[0]

    # 5. Explainable AI (Real contributions)
    coeffs = model.coef_
    intercept = model.intercept_

    age_contribution = coeffs[0] * age
    gender_contribution = coeffs[1] * gender_encoded
    education_contribution = coeffs[2] * education_encoded
    job_contribution = coeffs[3] * job_encoded
    experience_contribution = coeffs[4] * experience

    # 6. HTML Result
    result_html = f"""
    <div class='result' style="padding:20px;border-radius:10px;background:#f8f9fa;">

        <h2>💰 Predicted Salary</h2>

        <h1 style="color:green;">Rs. {prediction:,.2f}</h1>

        <hr>

        <h3>User Input</h3>
        <p>Age: {age}</p>
        <p>Experience: {experience}</p>
        <p>Gender: {gender}</p>
        <p>Education: {education}</p>
        <p>Job Title: {job_title}</p>

        <hr>

        <h3>Encoded Values</h3>
        <p>Gender Encoded: {gender_encoded}</p>
        <p>Education Encoded: {education_encoded}</p>
        <p>Job Encoded: {job_encoded}</p>

        <hr>

        <h3>Salary Breakdown (Model Understanding)</h3>

        <p>Base Salary: Rs. {intercept:,.2f}</p>

        <p>Age Contribution: Rs. {age_contribution:,.2f}</p>
        <p>Gender Contribution: Rs. {gender_contribution:,.2f}</p>
        <p>Education Contribution: Rs. {education_contribution:,.2f}</p>
        <p>Job Contribution: Rs. {job_contribution:,.2f}</p>
        <p>Experience Contribution: Rs. {experience_contribution:,.2f}</p>

        <hr>

        <h3>Final Output</h3>

        <p style="font-size:18px;">
            Salary = Base + All Contributions
        </p>

        <h2 style="color:green;">
            Rs. {prediction:,.2f}
        </h2>

    </div>
    """

    return render_template("index.html", result_html=result_html)


if __name__ == "__main__":
    app.run(debug=True)