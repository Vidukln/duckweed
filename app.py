import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Physics Analyzer", layout="wide")

st.title("🌿 Duckweed Physics-Aware Analyzer")
st.write("Upload pond images to estimate duckweed coverage with physics + uncertainty modeling")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = analyze_image(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(result["overlay"], caption="Segmentation + Flare Detection", use_container_width=True)
        st.image(result["uncertainty"], caption="Uncertainty Map", use_container_width=True)

    with col2:
        st.metric("🌿 Hard Area (m²)", f"{result['hard_area']:.6f}")
        st.metric("📈 Weighted Area (m²)", f"{result['weighted_area']:.6f}")
        st.metric("🌱 Blob Count", int(result["blobs"]))
        st.metric("⚠️ Mean Uncertainty", f"{result['mean_uncertainty']:.3f}")
        st.metric("🔥 Flare Ratio", f"{result['flare_ratio']:.3f}")
