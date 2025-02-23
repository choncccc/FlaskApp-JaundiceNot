from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import io

app = Flask(__name__)

model = tf.keras.models.load_model("JaundiceNotBVals.keras")

def extract_b_channel_features(image, bins=50):
    image = np.array(image.convert("RGB"))
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    b_channel = lab_image[:, :, 2]
    b_channel = (b_channel - b_channel.min()) / (b_channel.max() - b_channel.min())

    b_hist = np.histogram(b_channel, bins=bins, range=(0, 1), density=True)[0]

    return b_hist

def compute_way_kmeans_lab(image, k=6):
    image = np.array(image.convert("RGB"))
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    pixels = lab_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    normalized_B_values = ((cluster_centers[:, 2] - 128) / 127.0) * 50 + 20
    
    cluster_sizes = np.bincount(cluster_labels, minlength=k)
    total_pixels = np.sum(cluster_sizes)
    cluster_proportions = cluster_sizes / total_pixels
    
    WAY = np.dot(cluster_proportions, normalized_B_values)
    return WAY

def classify_jaundice(WAY):
    if 20 <= WAY < 25:
        return "Onset/Mild Jaundice"
    elif 25 <= WAY < 35:
        return "Moderate Jaundice"
    elif WAY >= 35:
        return "Severe Jaundice"
    return "Normal"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    features = extract_b_channel_features(image, bins=50).reshape(1, -1)

    if features.shape[1] != 50:
        return jsonify({'error': f'Invalid input shape. Expected 10, got {features.shape[1]}'}), 400

    prediction = model.predict(features)[0][0]
    print(prediction)
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
