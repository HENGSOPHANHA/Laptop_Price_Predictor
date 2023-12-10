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

    # Define the input fields for the features used in the model as selectboxes
    # Note: The options provided here are placeholders. Replace them with actual options from your dataset.
    company = st.selectbox('Select Company', ['HP', 'Dell', 'Lenovo', 'Asus', 'Apple', 'Acer', 'Microsoft', 'Toshiba', 'Other'])
    type_name = st.selectbox('Select Type', ['Ultrabook', 'Notebook', 'Gaming', '2-in-1 Convertible', 'Workstation', 'Netbook', 'Other'])
    inches = st.selectbox('Select Screen Size (in Inches)', ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
    screen_resolution = st.selectbox('Select Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', 'Other'])
    cpu = st.selectbox('Select CPU Model', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9', 'AMD Ryzen', 'Other'])
    ram = st.selectbox('Select RAM (GB)', ['4', '8', '16', '32', '64', 'Other'])
    memory = st.selectbox('Select Memory', ['128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD', '2TB SSD', 'Other'])
    gpu = st.selectbox('Select GPU Model', ['NVIDIA', 'AMD Radeon', 'Intel', 'Other'])
    op_sys = st.selectbox('Select Operating System', ['Windows 10', 'Windows 7', 'Linux', 'MacOS', 'Chrome OS', 'Other'])
    weight = st.selectbox('Select Weight (kg)', ['1', '1.5', '2', '2.5', '3', '3.5', '4', 'Other'])

    # Predict button
    if st.button('Predict Price'):
        # Prepare the feature array for prediction
        # Here, you might need to process these features (e.g., encoding) before prediction
        features = np.array([[company, type_name, inches, screen_resolution, cpu, ram, memory, gpu, op_sys, weight]])

        # Predicting the price
        predicted_price = model.predict(features)
        st.success(f'The predicted price of the laptop is approximately ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
