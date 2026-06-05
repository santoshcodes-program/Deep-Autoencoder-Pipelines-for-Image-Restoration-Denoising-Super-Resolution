import streamlit as st
from PIL import Image
import numpy as np
from UNetutils import remove_noise

st.title("Teeth X-Ray Denoising App")
st.write("Upload a noisy X-ray image and restore clean detail using a trained denoising model.")

uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "jpeg", "png"])
model_weights_path = "../models/unet_weights_2.pth"

def run_denoising(image_path):
    output_image = remove_noise(image_path, model_weights_path)
    return output_image

if uploaded_file is not None:
    # read image
    input_image = Image.open(uploaded_file).convert('L')
    
    st.subheader("Noisy Input")
    st.image(input_image, caption='Noisy X-ray', use_column_width=True)

    output_image = run_denoising(uploaded_file)
    output_image_pil = Image.fromarray(output_image)

    # display denoised image
    st.subheader("Denoised Image")
    st.image(output_image_pil, caption='Denoised Image', use_column_width=True)