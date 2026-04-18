import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Physics Analyzer", layout="wide")

st.title("🌿 Duckweed Physics-Aware Analyzer")

# =========================
# MODE SELECTION
# =========================
mode = st.radio("Select Mode", ["Single Image", "Multiple Images"])

# =========================
# SINGLE IMAGE MODE
# =========================
if mode == "Single Image":

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

        st.metric("Hard Area (m²)", f"{result['hard_area']:.6f}")
        st.metric("Weighted Area (m²)", f"{result['weighted_area']:.6f}")
        st.metric("Blob Count", int(result["blobs"]))
        st.metric("Flare Ratio", f"{result['flare_ratio']:.3f}")


# =========================
# MULTIPLE IMAGE MODE
# =========================
else:

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    results_summary = []

    if uploaded_files:

        for idx, uploaded_file in enumerate(uploaded_files):

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            result = analyze_image(img_rgb)

            st.subheader(f"Image {idx+1}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(img_rgb, caption="Original", use_container_width=True)

            with col2:
                st.image(result["overlay"], caption="Overlay", use_container_width=True)

            with col3:
                st.image(result["uncertainty"], caption="Uncertainty", use_container_width=True)

            results_summary.append([
                result["hard_area"],
                result["weighted_area"],
                result["blobs"],
                result["flare_ratio"]
            ])

        # =========================
        # SUMMARY ACROSS IMAGES
        # =========================
        results_summary = np.array(results_summary)

        st.divider()
        st.subheader("📊 Batch Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Hard Area", f"{np.mean(results_summary[:,0]):.6f}")

        with col2:
            st.metric("Avg Weighted Area", f"{np.mean(results_summary[:,1]):.6f}")

        with col3:
            st.metric("Blob Variance", f"{np.var(results_summary[:,2]):.3f}")

        with col4:
            st.metric("Avg Flare Ratio", f"{np.mean(results_summary[:,3]):.3f}")
