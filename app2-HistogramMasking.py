from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from sklearn.cluster import KMeans

app = Flask(__name__)

model = tf.keras.models.load_model("finalModelJaundiceNot.keras")  

import numpy as np
import cv2

def extract_yellow_histogram_features(image, bins=10, b_percentile=50, l_percentile=30, l_dark_threshold=20):
    image = np.array(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L, A, B = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    B_thresh = np.percentile(B, b_percentile)
    L_bright_thresh = np.percentile(L, l_percentile)
    L_dark_thresh = np.percentile(L, l_dark_threshold)

    yellow_mask = (B > B_thresh) & ((L > L_bright_thresh) | (L < L_dark_thresh))
    
    L_valid, A_valid, B_valid = L[yellow_mask], A[yellow_mask], B[yellow_mask]

    if L_valid.size == 0:
        return np.zeros(bins * 3)

    L_hist = np.histogram(L_valid / 255.0, bins=bins, range=(0, 1), density=True)[0]
    A_hist = np.histogram((A_valid + 128) / 255.0, bins=bins, range=(0, 1), density=True)[0]
    B_hist = np.histogram((B_valid + 128) / 255.0, bins=bins, range=(0, 1), density=True)[0]

    feature_vector = np.concatenate([L_hist, A_hist, B_hist])
    return feature_vector


import numpy as np
import cv2
from sklearn.cluster import KMeans

def compute_way_kmeans_lab(image, k=6, b_percentile=50, l_percentile=30):
    image = np.array(image.convert("RGB"))
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    L, A, B = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Compute thresholds for masking
    B_thresh = np.percentile(B, b_percentile)  # Yellow threshold
    L_thresh = np.percentile(L, l_percentile)  # Brightness threshold

    # Select only yellowish and bright pixels
    yellow_mask = (B > B_thresh) & (L > L_thresh)
    masked_pixels = lab_image[yellow_mask]

    if masked_pixels.shape[0] == 0:
        return 0  # Return 0 if no valid pixels are found to avoid errors

    # Apply K-means on masked pixels
    kmeans = KMeans(n_clusters=min(k, len(masked_pixels)), n_init=10, random_state=42)
    kmeans.fit(masked_pixels)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Convert B values to WAY formula
    normalized_B_values = ((cluster_centers[:, 2] - 128) / 127.0) * 50 + 20
    cluster_sizes = np.bincount(cluster_labels, minlength=k)

    total_pixels = np.sum(cluster_sizes)
    cluster_proportions = cluster_sizes / total_pixels
    WAY = np.dot(cluster_proportions, normalized_B_values)

    return WAY


def classify_jaundice(WAY):
    if 20 <= WAY <= 25:
        return "Onset/Mild Jaundice"
    elif 25 <= WAY < 35:
        return "Moderate Jaundice"
    elif WAY > 35:
        return "Severe Jaundice"
    else:
        return "No Jaundice"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    bins = 10  
    features = extract_yellow_histogram_features(image, bins).reshape(1, -1)
    expected_features = bins * 3

    if features.shape[1] != expected_features:
        return jsonify({'error': f'Invalid input shape. Expected {expected_features}, got {features.shape[1]}'}), 400
    
    prediction = model.predict(features)[0][0]
    predicted_label = 1 if prediction >= 0.50 else 0  
    response_data = {'prediction': str(predicted_label)}

    if predicted_label == 1:
        WAY = compute_way_kmeans_lab(image)
        severity = classify_jaundice(WAY)
        response_data["WAY"] = WAY
        response_data["severity"] = severity
    
    print(response_data)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)