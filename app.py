import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Physics Analyzer", layout="wide")

st.title("🌿 Duckweed Physics-Aware Analyzer")
st.write("Upload pond images to analyze duckweed coverage")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = analyze_image(img_rgb)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb, caption="Original Image", use_container_width=True)

    with col2:
        st.image(result["overlay"], caption="Segmentation + Flare", use_container_width=True)

    with col3:
        st.image(result["uncertainty"], caption="Uncertainty Map", use_container_width=True)

    st.divider()

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.metric("Hard Area (m²)", f"{result['hard_area']:.6f}")

    with col5:
        st.metric("Weighted Area (m²)", f"{result['weighted_area']:.6f}")

    with col6:
        st.metric("Blob Count", int(result["blobs"]))

    with col7:
        st.metric("Flare Ratio", f"{result['flare_ratio']:.3f}")
