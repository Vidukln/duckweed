#  Duckweed Physics-Aware Vision System

This project estimates duckweed coverage using:

- Gaussian Mixture Model (GMM) segmentation
- Optical flare detection (HSV + clustering)
- Camera geometry correction
- Uncertainty estimation
- Connected component analysis (biomass structure)

##  Deployment
This app is compatible with Streamlit Cloud.

##  Output Metrics
- Hard Area (pixel-based segmentation)
- Weighted Area (geometry-corrected biomass proxy)
- Blob count (structural growth indicator)
- Uncertainty map (confidence estimation)
- Flare ratio (lighting noise detection)

