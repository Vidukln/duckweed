import numpy as np
import cv2
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def analyze_image(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    # Predict clusters
    labels = model.predict(pixels)
    label_map = labels.reshape(h, w)

    # Identify green cluster (based on training means)
    means = model.means_
    green_idx = np.argmax(means[:, 1])  # green dominance

    # Mask
    green_mask = (label_map == green_idx)

    # Coverage
    coverage = np.sum(green_mask) / green_mask.size

    # Output image
    result = img.copy()
    result[green_mask] = [0, 255, 0]

    return result, float(coverage)
