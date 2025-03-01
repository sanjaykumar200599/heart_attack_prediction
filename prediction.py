import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler():
    """Load the trained heart attack model and pre-fitted scaler."""
    with open('heart_trained.sav', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)  # Load the pre-fitted scaler

    return model, scaler

def predict_heart_attack(features, model, scaler, feature_names):
    """Predict heart attack risk based on input features."""
    # Convert input to a Pandas DataFrame with correct column names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Apply the trained scaler (convert back to DataFrame with correct feature names)
    features_scaled = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)

    # Make prediction (pass DataFrame to preserve feature names)
    prediction = model.predict(features_scaled_df)
    probability = model.predict_proba(features_scaled_df)[:, 1]

    return prediction[0], probability[0]

# Streamlit UI
st.title("Heart Attack Prediction App")

# Load model and scaler
model, scaler = load_model_and_scaler()

# Define the column names (must match training data)
feature_names = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "caa"]

# Collect user inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.radio("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])

# Create feature array
input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, ca]

# Predict button
if st.button("Predict"): 
    # Ensure input feature count matches expected feature names
    assert len(input_features) == len(feature_names), "Feature count mismatch!"
    
    # Make prediction
    prediction, probability = predict_heart_attack(input_features, model, scaler, feature_names)
    
    # Display result
    if prediction == 1:
        st.error(f"High risk of heart attack! Probability: {probability:.2f}")
    else:
        st.success(f"Low risk of heart attack. Probability: {probability:.2f}")
