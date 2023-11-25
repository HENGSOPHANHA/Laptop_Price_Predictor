import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Assuming df is your DataFrame
df = pd.read_csv("your_data.csv")

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('Ram(in GB)', df['Ram'].unique())
os = st.selectbox('OS', df['OpSys'].unique())
weight = st.number_input('Weight of the laptop')
touchscreen = st.selectbox('Touchscreen', df['TouchScreen'].unique())
ips = st.selectbox('IPS', df['IPS'].unique())
screen_size = st.number_input('Screen Size')
resolution = st.number_input('Screen Resolution (PPI)')
cpu = st.selectbox('CPU', df['CPU_name'].unique())
hdd = st.selectbox('HDD(in GB)', df['HDD'].unique())
ssd = st.selectbox('SSD(in GB)', df['SSD'].unique())
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

if st.button('Predict Price'):
    ppi = resolution / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'OpSys': [os],
        'Weight': [weight],
        'TouchScreen': [touchscreen],
        'IPS': [ips],
        'PPI': [ppi],
        'CPU_name': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu]
    })

    # Perform one-hot encoding for categorical variables
    query = pd.get_dummies(query, columns=['Company', 'TypeName', 'CPU_name', 'OpSys', 'Gpu brand'])

    # Ensure the order of columns matches the order during training
    query = query[df.drop('Predicted Price', axis=1).columns]

    # Make prediction
    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("Predicted price for this laptop could be between " +
             str(prediction-100) + "$" + " to " + str(prediction+100) + "$")
