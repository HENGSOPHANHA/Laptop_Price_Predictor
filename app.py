import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Correct path to your saved model
model_path = 'laptop_price_predictor.pkl'  # Just pass the file path as a string

# Load the trained pipeline
pipeline = joblib.load(model_path)  # This should now work correctly

# Streamlit app
def run():
    st.title('Laptop Price Predictor')

    # Input fields
    company = st.selectbox('Company', ['Dell', 'Apple', 'HP', 'Lenovo', 'Acer', 'Asus', 'Other'])
    type_name = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Gaming', 'Business', 'Other'])
    inches = st.number_input('Screen Size in Inches', min_value=10.0, max_value=18.0, value=15.6, step=0.1)
    touchscreen = st.radio('Touchscreen', ['Yes', 'No'])
    ips = st.radio('IPS Panel', ['Yes', 'No'])
    ppi = st.number_input('Pixels Per Inch', min_value=100, max_value=300, value=141, step=1)
    cpu_name = st.selectbox('CPU Name', ['Intel Core i7', 'Intel Core i5', 'AMD Ryzen', 'Other'])
    ram = st.slider('RAM in GB', min_value=4, max_value=64, value=16)
    weight = st.number_input('Weight in Kg', min_value=0.5, max_value=5.0, value=1.8, step=0.1)
    opsys_simple = st.selectbox('Operating System', ['Windows', 'Linux', 'Apple', 'Other'])
    hdd = st.number_input('HDD Size in GB (if any)', min_value=0, max_value=2000, value=0, step=1)
    ssd = st.number_input('SSD Size in GB (if any)', min_value=0, max_value=2000, value=512, step=1)

    # Convert Yes/No to 1/0
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    if st.button('Predict Price'):
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Inches': [inches],
            'Touchscreen': [touchscreen],
            'IPS': [ips],
            'PPI': [ppi],
            'CPU_Name': [cpu_name],
            'Ram': [ram],
            'Weight': [weight],
            'OpSys_Simple': [opsys_simple],
            'HDD': [hdd],
            'SSD': [ssd]
        })

        # Predict the price
        predicted_log_price = pipeline.predict(input_data)
        predicted_price = np.exp(predicted_log_price)

        st.success(f"Predicted Laptop Price: ${predicted_price[0]:.2f}")

if __name__ == '__main__':
    run()
