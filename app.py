import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv("traineddata.csv")

st.title("Laptop Price Predictor")

company_mapping = {'Apple': 0, 'Dell': 1, 'HP': 2, 'Acer': 3, 'Lenovo': 4, 'Asus': 5, 'MSI': 6, 'Toshiba': 7, 'Razer': 8, 'Mediacom': 9, 'Chuwi': 10}
type_mapping = {'Ultrabook': 0, 'Notebook': 1, 'Netbook': 2, 'Gaming': 3, '2 in 1 Convertible': 4, 'Workstation': 5}
os_mapping = {'Mac': 0, 'Windows': 1, 'Linux': 2, 'Chrome OS': 3, 'No OS': 4}
cpu_mapping = {'Intel Core i5': 0, 'Intel Core i7': 1, 'Intel Core i3': 2, 'Intel Celeron Dual Core': 3, 'Intel Core i6': 4}

company = st.selectbox('Brand', data['Company'].unique(), format_func=lambda x: company_mapping.get(x, x))
type = st.selectbox('Type', data['TypeName'].unique(), format_func=lambda x: type_mapping[x])
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('OS', data['OpSys'].unique(), format_func=lambda x: os_mapping[x])
weight = st.number_input('Weight of the laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['CPU_name'].unique(), format_func=lambda x: cpu_mapping[x])
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_resolution = float(resolution.split('x')[0])
    Y_resolution = float(resolution.split('x')[1])

    ppi = ((X_resolution**2) + (Y_resolution**2))**0.5 / (screen_size)

    query = np.array([company_mapping.get(company, company), type_mapping[type], ram, weight,
                      touchscreen, ips, ppi, cpu_mapping[cpu], hdd, ssd, gpu, os_mapping[os]])

    query = query.reshape(1, 12)
    prediction_value = rf.predict(query)[0]
    prediction = float(np.exp(prediction_value))

    st.title("Predicted price for this laptop could be between " +
             str(prediction - 10) + "$" + " to " + str(prediction + 10) + "$")
