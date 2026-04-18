import streamlit as st
import numpy as np
import cv2
from model import analyze_image

st.set_page_config(page_title="Duckweed Physics Analyzer", layout="wide")

st.title("🌿 Duckweed Physics-Aware Analyzer")

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

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Hard Area (m²)", f"{result['hard_area']:.6f}")
        c2.metric("Weighted Area (m²)", f"{result['weighted_area']:.6f}")
        c3.metric("Blobs", int(result["blobs"]))
        c4.metric("Flare Ratio", f"{result['flare_ratio']:.3f}")


# =========================
# MULTIPLE IMAGE MODE
# =========================
else:

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        all_results = []

        for idx, uploaded_file in enumerate(uploaded_files):

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            result = analyze_image(img_rgb)

            st.markdown(f"## 🌿 Image {idx + 1}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(img_rgb, caption="Original Image", use_container_width=True)

            with col2:
                st.image(result["overlay"], caption="Segmentation + Flare", use_container_width=True)

            with col3:
                st.image(result["uncertainty"], caption="Uncertainty Map", use_container_width=True)

            # =========================
            # PER IMAGE METRICS (FIXED)
            # =========================
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Hard Area (m²)", f"{result['hard_area']:.6f}")
            m2.metric("Weighted Area (m²)", f"{result['weighted_area']:.6f}")
            m3.metric("Blobs", int(result["blobs"]))
            m4.metric("Flare Ratio", f"{result['flare_ratio']:.3f}")

            st.divider()

            all_results.append([
                result["hard_area"],
                result["weighted_area"],
                result["blobs"],
                result["flare_ratio"]
            ])

        # =========================
        # SUMMARY ACROSS ALL IMAGES
        # =========================
        all_results = np.array(all_results)

        st.subheader("📊 Batch Summary")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("Avg Hard Area", f"{np.mean(all_results[:,0]):.6f}")
        s2.metric("Avg Weighted Area", f"{np.mean(all_results[:,1]):.6f}")
        s3.metric("Blob Variance", f"{np.var(all_results[:,2]):.3f}")
        s4.metric("Avg Flare Ratio", f"{np.mean(all_results[:,3]):.3f}")
