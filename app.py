import streamlit as st
import base64
from predictor import predict
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-repeat: repeat;
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

def header_white_bg(text, fontsize = 40, bold = True):
    st.markdown(
        f"""
        <span style="background:rgba(255, 255, 255, 0.8); font-size:{fontsize}px; font-weight:{"bold" if bold else "normal"}">{text}</span>
        """,
        unsafe_allow_html=True
    )

def diagnose_health(file):
    prediction = predict(file)
    predicted_strings = []
    for p in prediction:
        predicted_string = f"{p['predicted']}, Probability: {float(p['probability']):.2f}"
        predicted_strings.append(predicted_string)
    return predicted_strings

def app():
    add_bg_from_local('assets/background.png')
    header_white_bg(f'<span style="color:green">Plant</span><span style="color:orange">Dx</span><span style="color:green">: Diagnosis in a Snap!</span> ')

    # Upload image of plant
    header_white_bg("Upload an image of your plant:", fontsize=32)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        header_white_bg("Preview of the selected image:", fontsize=28, bold=False)
        st.image(uploaded_file)

    # Get diagnosis button
    if st.button("Get Diagnosis"):
        if uploaded_file is not None:
            # Diagnose plant health and display results
            result = diagnose_health(uploaded_file)
            st.success(f"Your plant is {result}")
        else:
            st.warning("Please upload an image of your plant first")

    # Create user profile button
    if st.button("Create User Profile"):
        st.subheader("User Profile")
        # Prompt user to add their name and the plant they own
        user_name = st.text_input("Enter your name:")
        plant_name = st.text_input("Enter the plant you own:")
        if user_name and plant_name:
            st.success(f"User profile created for {user_name} with plant {plant_name}")


# Run Streamlit app
if __name__ == "__main__":
    app()
