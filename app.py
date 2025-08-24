import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load('LR_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Streamlit app
st.title("Heart Stroke Prediction by Kartik ")
st.markdown("Provide the following details")

age = st.slider('Age', 18, 100, 40)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure(mm Hg)', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
max_heart_rate = st.slider('Maximum Heart Rate Achieved', min_value=60, max_value=200, value=120)
exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('Oldpeak', min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    raw_input = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': 1 if fasting_bs == 'Yes' else 0,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': 1 if exercise_angina == 'Yes' else 0,
        'oldpeak': oldpeak,
        'chest_pain': chest_pain,
        'resting_ecg': resting_ecg,
        'st_slope': st_slope
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # One-hot encode categorical vars
    input_df = pd.get_dummies(input_df)

    # Add missing columns (so it matches training dataset)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("❌ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
