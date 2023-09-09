# app.py
import streamlit as st
import cv2

st.title("OpenCV and Streamlit Example")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = cv2.imread(uploaded_image.name)
    st.image(image, caption="Uploaded Image", use_column_width=True)
