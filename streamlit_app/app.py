import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/salary_model.pkl")

st.title("Employee Salary Prediction")

st.write("Enter employee details below.")

# Numerical Inputs
age = st.number_input("Age", 18, 100, 30)
fnlwgt = st.number_input("Final Weight", 10000, 1000000, 50000)
educational_num = st.number_input("Educational Number", 1, 20, 10)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 100, 40)

# Dropdown Inputs (encoded internally)
workclass = st.selectbox(
    "Workclass",
    ["Private", "Government", "Self-Employed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Never-married", "Married", "Divorced"]
)

occupation = st.selectbox(
    "Occupation",
    ["Tech-support", "Sales", "Exec-managerial"]
)

relationship = st.selectbox(
    "Relationship",
    ["Not-in-family", "Husband", "Wife"]
)

race = st.selectbox(
    "Race",
    ["White", "Black", "Asian"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

native_country = st.selectbox(
    "Native Country",
    ["United-States", "India", "Canada"]
)

# Simple manual encoding
workclass_map = {
    "Private": 1,
    "Government": 2,
    "Self-Employed": 3
}

marital_map = {
    "Never-married": 1,
    "Married": 2,
    "Divorced": 3
}

occupation_map = {
    "Tech-support": 1,
    "Sales": 2,
    "Exec-managerial": 3
}

relationship_map = {
    "Not-in-family": 1,
    "Husband": 2,
    "Wife": 3
}

race_map = {
    "White": 1,
    "Black": 2,
    "Asian": 3
}

gender_map = {
    "Male": 1,
    "Female": 0
}

country_map = {
    "United-States": 1,
    "India": 2,
    "Canada": 3
}

if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass_map[workclass]],
        'fnlwgt': [fnlwgt],
        'educational-num': [educational_num],
        'marital-status': [marital_map[marital_status]],
        'occupation': [occupation_map[occupation]],
        'relationship': [relationship_map[relationship]],
        'race': [race_map[race]],
        'gender': [gender_map[gender]],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [country_map[native_country]]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Predicted Salary: >50K")
    else:
        st.success("Predicted Salary: <=50K")