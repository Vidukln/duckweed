import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Analyzer", layout="centered")

st.title("🌿 Duckweed Coverage Analyzer")
st.write("Upload pond images to estimate duckweed coverage")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read image. Please upload a valid file.")
    else:
        result, coverage = analyze_image(img)

        st.image(result, caption="Detected Duckweed", use_container_width=True)

        st.metric("🌿 Coverage", f"{coverage*100:.2f}%")
