
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache
def load_model():
    with open('laptoppricepredictor.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit application
def main():
    st.title('Laptop Price Predictor')

    # Define the input fields for the features used in the model
    # Note: You need to adjust these input fields based on your model's features
    company = st.selectbox('Company', ['HP', 'Dell', 'Lenovo', 'Asus', ...])
    type_name = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Gaming', ...])
    inches = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, step=0.1)
    # Continue for all features...

    # Predict button
    if st.button('Predict Price'):
        # Prepare the feature array for prediction (based on the model's requirement)
        features = np.array([[company, type_name, inches, ...]])
        # You might need to process these features before prediction
        # e.g., encoding categorical variables

        # Predicting the price
        predicted_price = model.predict(features)
        st.success(f'The predicted price of the laptop is ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
