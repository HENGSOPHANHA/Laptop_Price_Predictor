import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Use pd.read_pickle instead of pickle.load for loading DataFrame
df = pd.read_pickle('df.pkl')

st.title("Laptop Predictor")

# Collect user inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['CPU_name'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['OpSys'].unique())

# Button to trigger prediction
if st.button('Predict Price'):
    # Create a DataFrame with user inputs
    user_input = pd.DataFrame([[company, type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os]],
                               columns=df.columns)

    # Concatenate user input with original DataFrame
    input_df = pd.concat([df, user_input], ignore_index=True)

    # Use get_dummies to one-hot encode categorical variables
    input_df_encoded = pd.get_dummies(input_df, columns=['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'OpSys'])

    # Select the last row (user input) for prediction
    user_input_encoded = input_df_encoded.iloc[[-1]]

    # Use reshape(1, -1) instead of reshape(1, 12)
    user_input_encoded = user_input_encoded.values.reshape(1, -1)

    try:
        predicted_price = int(np.exp(pipe.predict(user_input_encoded)[0]))
        st.title("The predicted price of this configuration is $" + str(predicted_price))
    except ValueError as e:
        st.error(f"Error predicting price: {e}")
