import streamlit as st
import pandas as pd
import sys
import os

# Add src to path to import predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import make_prediction

# --- Page Config ---
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("💰 Employee Salary Prediction")
st.markdown("Enter employee details below to predict if their income exceeds **$50,000/year**.")
st.divider()

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=75, value=30)
        workclass = st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 
            'Self-emp-inc', 'Federal-gov', 'Others'
        ])
        fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)
        edu_num = st.slider("Educational Num", min_value=1, max_value=16, value=10)
        marital = st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Never-married', 'Divorced', 
            'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
        ])
        occupation = st.selectbox("Occupation", [
            'Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical',
            'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving',
            'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv',
            'Priv-house-serv', 'Armed-Forces', 'Others'
        ])
        
    with col2:
        relationship = st.selectbox("Relationship", [
            'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'
        ])
        race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.radio("Gender", ['Male', 'Female'])
        cap_gain = st.number_input("Capital Gain", min_value=0, value=0)
        cap_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
        country = st.text_input("Native Country", value="United-States")

    submit = st.form_submit_button("🚀 Predict Salary Class")

# --- Logic & Output ---
if submit:
    # Prepare input dictionary
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': edu_num,
        'marital-status': marital,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': cap_gain,
        'capital-loss': cap_loss,
        'hours-per-week': hours,
        'native-country': country
    }
    
    try:
        # Paths relative to the app.py location
        model_p = os.path.join(os.path.dirname(__file__), '..', 'models', 'salary_model.pkl')
        encoder_p = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')
        
        prediction, probability = make_prediction(input_data, model_p, encoder_p)
        
        if prediction == 1:
            st.markdown(f"""
                <div class="result-box success">
                    Prediction: >$50,000 / Year<br>
                    <span style="font-size: 16px;">Confidence: {probability*100:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="result-box warning">
                    Prediction: ≤$50,000 / Year<br>
                    <span style="font-size: 16px;">Confidence: {probability*100:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Ensure the model is trained and 'models/salary_model.pkl' exists.")

# --- Footer ---
st.divider()
st.caption("ML System Design Course Project | Developed by Antigravity AI")
