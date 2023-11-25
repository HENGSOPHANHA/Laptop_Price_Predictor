import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import pickle
from PIL import Image

# Cache data loading
@st.cache  
def load_data():
    data = pd.read_csv("traineddata.csv")
    return data

# Cache model
@st.cache(allow_output_mutation=True)
def load_model():
    file = open('pipe.pkl','rb') 
    model = pickle.load(file)
    return model

data = load_data()
rf_model = load_model() 

# Title 
st.title("Laptop Price Predictor")

# Inputs
laptop_brand = st.selectbox('Brand', data['Company'].unique())
laptop_type =  st.selectbox('Type', data['TypeName'].unique())  

# Add other inputs

# Create features dataframe
X = pd.DataFrame(data=[[laptop_brand, laptop_type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]]) 

# Validate inputs
if st.button("Predict"):
   
   # Predict  
   prediction = rf_model.predict(X)
   prediction = round(np.exp(prediction[0]))
   
   # Display 
   st.header("Predicted price: ${}".format(prediction))
