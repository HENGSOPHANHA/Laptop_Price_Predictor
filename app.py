import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', data['Company'].unique())

# type of laptop

type = st.selectbox('Type', data['TypeName'].unique())

# Ram present in laptop

ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# os of laptop

os = st.selectbox('OS', data['OpSys'].unique())

# weight of laptop

weight = st.number_input('Weight of the laptop')

# touchscreen available in laptop or not

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS

ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size

screen_size = st.number_input('Screen Size')

# resolution of laptop

resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu

cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

#..............

# Assuming 'company', 'type', 'os', and 'cpu' are categorical variables
# You should replace this with the actual encoding methods you used during training

company_encoded = pd.get_dummies(data['Company'], drop_first=True)
type_encoded = pd.get_dummies(data['TypeName'], drop_first=True)
os_encoded = pd.get_dummies(data['OpSys'], drop_first=True)
cpu_encoded = pd.get_dummies(data['CPU_name'], drop_first=True)

# Create a dictionary to map category values to numerical values
company_mapping = dict(zip(data['Company'].unique(), range(len(data['Company'].unique()))))
type_mapping = dict(zip(data['TypeName'].unique(), range(len(data['TypeName'].unique()))))
os_mapping = dict(zip(data['OpSys'].unique(), range(len(data['OpSys'].unique()))))
cpu_mapping = dict(zip(data['CPU_name'].unique(), range(len(data['CPU_name'].unique()))))

# Replace category values with numerical values in the query array
query[0] = [company_mapping.get(query[0][0], query[0][0]),
            type_mapping.get(query[0][1], query[0][1]),
            query[0][2],
            query[0][3],
            query[0][4],
            query[0][5],
            query[0][6],
            cpu_mapping.get(query[0][7], query[0][7]),
            query[0][8],
            query[0][9],
            query[0][10],
            os_mapping.get(query[0][11], query[0][11])]
#.................

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

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    prediction_value = np.array(rf.predict(query))[0]  # Convert to NumPy array
    prediction = float(np.exp(prediction_value))

    st.title("Predicted price for this laptop could be between " +
             str(prediction - 10) + "$" + " to " + str(prediction + 10) + "$")
