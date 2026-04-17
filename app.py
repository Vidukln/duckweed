import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Analyzer")

st.title("🌿 Duckweed Coverage Analyzer")
st.write("Upload pond images to estimate duckweed coverage")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Run model
    result, coverage = analyze_image(img)

    # Show results
    st.image(result, caption="Detected Duckweed", use_column_width=True)
    st.write(f"🌿 Coverage: {coverage:.4f}")
