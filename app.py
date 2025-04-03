# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:56:09 2025

@author: gowtham.balachan
"""
##This is simple comment
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
#This is for the commit
# Load the XGBoost model
with open('./Model Store/xgboost', 'rb') as model_file:
    model = pickle.load(model_file)

# Title and instructions
st.title("🚢 Titanic Survival Prediction App test v1")
st.markdown("### Enter Passenger Details to Predict Survival")

# Input fields for prediction
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.number_input('Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare Amount', 0.0, 1000.0, 50.0)
embarked = st.selectbox('Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

# Encode categorical features
sex_encoded = 0 if sex == 'male' else 1
embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Prepare the input data
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Prediction logic
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived 😎" if prediction[0] == 1 else "Did Not Survive 😢"
    st.success(f"**Prediction:** {result}")
