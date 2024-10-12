import joblib
import numpy as np
import streamlit as st
import pandas as pd


class Prediction:
    def __init__(self):
        self.data = pd.read_csv("dataframe7.csv")
        self.scaler = joblib.load("scaler7.model")
        self.labeller = joblib.load("labeller7.model")
        self.model = joblib.load("model7.model")
        self.numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender Encoded']
        self.X = self.data[self.numerical_columns]

    def cluster_predict(self):
        try:
            gender = st.selectbox(
                "sex",
                [] + ["Male", "Female"])
            age = int(st.text_input("Age", "0"))
            income = int(st.text_input("Income", "0"))
            spending_score = int(st.text_input("Spending Score", "0"))
        except ValueError:
            st.warning("Please input value for all fields")
            return None
        if not (gender, age, income, spending_score):
            st.warning("All inputs must be filled with valid inputs")
        col1, col2, col3 = st.columns([1, 0.6, 1])
        with col2:
            if st.button("predict"):
                gender = self.labeller.transform([gender])

                # Prepare the data for scaling and prediction as a DataFrame
                input_data = pd.DataFrame({
                    'Gender Encoded': [gender[0]],  # gender has already been encoded
                    'Age': [age],
                    'Annual Income (k$)': [income/1000],
                    'Spending Score (1-100)': [spending_score]
                })

                # Transform the input data
                scaled_data = self.scaler.transform(input_data[self.numerical_columns])

                st.success(f"cluster = {self.model.predict(scaled_data)[0]}")
                return
