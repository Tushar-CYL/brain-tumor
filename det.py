import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

# Load the trained model
model = joblib.load('svm_model.pkl')

# Define the classes
classes = {0: 'Negetive Tumor', 1: 'Positive Tumor'}

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = img.reshape(1, -1) / 255
    return img

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to check for brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    img = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Display the image and prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: **{classes[prediction[0]]}**")
