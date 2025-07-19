import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Salary Prediction App")

# User input
age = st.number_input("Age", min_value=18, max_value=65, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", [
    "Software Engineer", "Data Scientist", "Product Manager",
    "Designer", "Accountant", "Administrative Assistant"
])
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0)

# Predict button
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ₹{prediction:,.2f}")