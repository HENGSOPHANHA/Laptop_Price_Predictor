import streamlit as st
import pandas as pd 
import numpy as np
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

# Page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon=":computer:")

# Title 
st.title("Laptop Price Predictor")

# Image 
img = Image.open('laptop.jpg')
st.image(img, width=300)

# Inputs
laptop_brand = st.selectbox('Brand', data['Company'].unique())
laptop_type =  st.selectbox('Type', data['TypeName'].unique())  

# Add other inputs

# Create features dataframe
X = pd.DataFrame(data=[[laptop_brand, laptop_type, ..., gpu, os]]) 

# Validate inputs
if st.button("Predict"):
   
   # Predict  
   prediction = rf_model.predict(X)
   prediction = round(np.exp(prediction[0]))
   
   # Display 
   st.header("Predicted price: ${}".format(prediction))
