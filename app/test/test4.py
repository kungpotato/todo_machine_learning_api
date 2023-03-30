
# pip install tensorflow
# pip install opencv-python
# pip install pillow
# pip install matplotlib

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

# Download and load the COCO dataset labels
LABELS_URL = "https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt"
os.system(f"wget -q -nc {LABELS_URL}")
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# Download the pre-trained model
MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
os.system(f"wget -q -nc {MODEL_URL} && tar -xzf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz")

# Load the pre-trained model
model = tf.saved_model.load('ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')

# Function to preprocess the input image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(320, 320))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict objects in the image
def predict_objects(image_path):
    image = preprocess_image(image_path)
    detections = model(image)
    return detections

# Function to visualize the detections
def visualize_detections(image_path, detections, threshold=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    for box, score, class_id in zip(boxes, scores, classes):
        if score > threshold:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{labels[int(class_id)]}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()

# Predict and visualize objects in a sample image
sample_image_path = "sample.jpg"  # Replace with your image path
detections = predict_objects(sample_image_path)
visualize_detections(sample_image_path, detections, threshold=0.5)
