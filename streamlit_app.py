import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import gdown
from PIL import Image

@st.cache_resource
def load_model():
    file_id = "15hIWAY8rcQuP3LfTK7FLEktca2-vPSi7"  # ⬅️ Replace with your actual Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "siamese_model.h5"
    gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output, compile=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = load_model()

# Preprocessing function
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    return img.reshape(1, 128, 128, 3)

st.title("Facial Identity Verification")
st.write("Upload a selfie and an ID image to verify identity.")

img1_file = st.file_uploader("Upload Selfie Image", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload ID Image", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    st.image([img1, img2], caption=["Selfie", "ID Image"], width=150)

    img1_preprocessed = preprocess_image(img1)
    img2_preprocessed = preprocess_image(img2)

    pred = model.predict([img1_preprocessed, img2_preprocessed])[0][0]
    threshold = 0.5

    st.markdown("### Result:")
    if pred >= threshold:
        st.success(f"✅ Match! (Score: {pred:.2f})")
    else:
        st.error(f"❌ Not a Match. (Score: {pred:.2f})")
