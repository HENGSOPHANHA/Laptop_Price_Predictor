
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

    # Input fields based on the features used in the model
    # Note: These should be adjusted to match the exact features of your model
    company = st.selectbox('Company', ['HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Apple', 'Samsung', 'Razer', 'Microsoft', 'Xiaomi'])
    type_name = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'])
    inches = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, step=0.1, format="%.1f")
    screen_resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1440', '1440x900', '2560x1600', '2304x1440'])
    cpu = st.text_input('CPU Model (e.g., Intel Core i7)')
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    memory = st.text_input('Memory (e.g., 512GB SSD)')
    gpu = st.text_input('GPU Model (e.g., NVIDIA GeForce GTX 1050)')
    op_sys = st.selectbox('Operating System', ['Windows 10', 'Windows 7', 'Linux', 'Mac OS', 'Chrome OS', 'Windows 10 S', 'No OS', 'Windows 8'])
    weight = st.number_input('Weight (in kg)', min_value=0.5, max_value=5.0, step=0.1, format="%.1f")

    # Predict button
    if st.button('Predict Price'):
        # Prepare the feature array for prediction
        # Note: This part needs to be customized based on how the model expects the input features
        features = np.array([[company, type_name, inches, screen_resolution, cpu, ram, memory, gpu, op_sys, weight]])
        
        # Feature preprocessing (if required by the model)
        # Example: Encoding categorical variables, feature scaling, etc.

        # Predicting the price
        predicted_price = model.predict(features)
        st.success(f'The predicted price of the laptop is approximately ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
