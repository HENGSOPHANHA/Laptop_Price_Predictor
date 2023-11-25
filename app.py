import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Use pd.read_pickle instead of pickle.load for loading DataFrame
df = pd.read_pickle('df.pkl')

# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

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

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    # Create a DataFrame with the input data for transformation
    input_data = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'TouchScreen': touchscreen,
        'IPS': ips,
        'PPI': ppi,
        'CPU_name': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'OpSys': os
    }])

    # Use pipe.predict instead of pipe.predict
    prediction = int(np.exp(pipe.named_steps['xgbregressor'].predict(input_data)[0]))

    st.title("Predicted price for this laptop could be between " +
             str(prediction-100)+"$" + " to " + str(prediction+100)+"$")
