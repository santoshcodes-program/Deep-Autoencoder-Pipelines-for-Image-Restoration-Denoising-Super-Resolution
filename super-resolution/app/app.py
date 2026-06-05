import streamlit as st
from PIL import Image
import numpy as np
import torch
from ModelUtils import super_resolve_image

st.title("Image Super Resolution App")
st.write("Upload a low-resolution image and enhance details using a trained super-resolution model.")

uploaded_file = st.file_uploader("Choose a low-quality image...", type=["jpg", "jpeg", "png"])

model_weights_path = "../weights/SuperResWeights.pth"

def run_super_resolution(image_path):
    output_image = super_resolve_image(image_path, model_weights_path)
    return output_image

if uploaded_file is not None:
    # read image
    input_image = Image.open(uploaded_file).convert('RGB')
    
    st.subheader("Input Image")
    st.image(input_image, caption='Input Image', width=256)

    output_image = run_super_resolution(uploaded_file)
    normalized_output = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_uint8 = (normalized_output * 255).astype(np.uint8)

    output_image_pil = Image.fromarray(np.transpose(output_uint8, (1, 2, 0)))

    # Display denoised image
    st.subheader("Enhanced Image")
    st.image(output_image_pil, caption='Enhanced Image', width=256)