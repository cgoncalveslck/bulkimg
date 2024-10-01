from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import os
import torch

app = Flask(__name__)
CORS(app)

# Load a pre-trained EfficientNet model
efficient_net = models.efficientnet_b0(pretrained=True)
efficient_net.eval()

# Define the transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = efficient_net(image).cpu().numpy()
    return embedding

@app.route('/photos/<filename>')
def get_photo(filename):
    return send_from_directory('photos', filename)

@app.route('/images')
def get_images():
    image_folder = './photos'
    image_files = os.listdir(image_folder)

    # Calculate embeddings for all images
    embeddings = np.array([get_image_embedding(os.path.join(image_folder, img)).flatten() for img in image_files])

    # Clustering images using K-Means
    num_clusters = min(len(image_files), 10)  # You can adjust the number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Organize images by their cluster
    clustered_images = {i: [] for i in range(num_clusters)}
    for img, label in zip(image_files, labels):
        clustered_images[label].append(img)

    # Flatten the clustered images for ordered display
    ordered_images = []
    for cluster in clustered_images.values():
        ordered_images.extend(cluster)

    return jsonify(ordered_images)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
