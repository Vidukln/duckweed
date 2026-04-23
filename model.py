import numpy as np
import cv2
import pickle
from sklearn.mixture import GaussianMixture

#LOAD MODEL
with open("gmm_model.pkl", "rb") as f:
    model = pickle.load(f)

LENS_TO_WATER_MM = 297
ROI_SIZE_MM = 183

L = LENS_TO_WATER_MM / 1000.0
ROI = ROI_SIZE_MM / 1000.0


#FLARE DETECTION
def detect_flare(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    bright = V > 220
    low_sat = S < 60

    return bright & low_sat

def camera_projection_model(h, w):
    fx = w / ROI
    fy = h / ROI

    cx, cy = w / 2, h / 2

    y, x = np.indices((h, w))

    X = (x - cx) / fx
    Y = (y - cy) / fy

    Z = L

    r = np.sqrt(X**2 + Y**2)
    cos_theta = Z / np.sqrt(Z**2 + r**2)

    return np.clip(cos_theta, 0.2, 1.0)

def analyze_image(image):

    img = image.copy()
    h, w = img.shape[:2]

    pixels = img.reshape(-1, 3)

    # GMM prediction
    labels = model.predict(pixels)
    probs = model.predict_proba(pixels)

    label_map = labels.reshape(h, w)
    prob_map = probs.reshape(h, w, 3)

    means = model.means_

    def green_score(m):
        R, G, B = m
        return G - 0.5 * (R + B)

    green_idx = np.argmax([green_score(m) for m in means])
    black_idx = np.argmin([np.mean(m) for m in means])
    flare_idx = np.argmax([np.mean(m) for m in means])

    green_mask = (label_map == green_idx)
    black_mask = (label_map == black_idx)
    flare_mask_gmm = (label_map == flare_idx)

    flare_mask_optical = detect_flare(img)
    flare_mask = flare_mask_gmm | flare_mask_optical

    W = prob_map[:, :, green_idx]
    W = W / (np.max(W) + 1e-8)

    #Geometry correction
    cos_theta = camera_projection_model(h, w)
    W_geo = W / cos_theta
    W_geo = W_geo / (np.max(W_geo) + 1e-8)

    uncertainty = 1 - np.max(prob_map, axis=2)

    ROI_area = ROI * ROI
    pixel_area = ROI_area / (h * w)

    hard_area = np.sum(green_mask) * pixel_area
    weighted_area = np.sum(W_geo) * pixel_area

    num_labels, _ = cv2.connectedComponents(green_mask.astype(np.uint8))
    blobs = num_labels - 1

    overlay = img.copy()

    overlay[green_mask] = [0, 255, 0]
    overlay[black_mask] = [40, 40, 40]
    overlay[flare_mask] = [255, 255, 0]

    return {
        "overlay": overlay,
        "uncertainty": uncertainty,
        "hard_area": float(hard_area),
        "weighted_area": float(weighted_area),
        "blobs": int(blobs),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "flare_ratio": float(np.mean(flare_mask))
    }
