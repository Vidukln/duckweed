import numpy as np
import cv2
import pickle

with open("gmm_model.pkl", "rb") as f:
    model = pickle.load(f)


def analyze_image(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    labels = model.predict(pixels)
    label_map = labels.reshape(h, w)

    means = model.means_

    green_idx = np.argmax(means[:, 1])

    mask = (label_map == green_idx)

    coverage = np.sum(mask) / mask.size

    result = img.copy()
    result[mask] = [0, 255, 0]

    return result, float(coverage)
