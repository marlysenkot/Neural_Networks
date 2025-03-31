import tensorflow as tf
import numpy as np
import cv2
import sys

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Load and preprocess an image for model prediction."""
    image = cv2.imread(image_path)  # Read image
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        sys.exit(1)
    image = cv2.resize(image, (30, 30))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(model, image_path):
    """Predict the class of a given image."""
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = np.max(prediction)  # Get confidence level
    return predicted_class, confidence

# Mapping of class numbers to labels (Modify this according to your dataset)
class_labels = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Overtaking",
    10: "No Overtaking for Trucks",
    11: "Right of Way at Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "No Entry for Trucks",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrian Crossing",
    28: "Children Crossing",
    29: "Bicycle Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Overtaking",
    42: "End of No Overtaking for Trucks"
}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    model = load_model(model_path)
    predicted_class, confidence = predict(model, image_path)
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Traffic Sign: {class_labels.get(predicted_class, 'Unknown')}")
    print(f"Confidence: {confidence:.2f}")
