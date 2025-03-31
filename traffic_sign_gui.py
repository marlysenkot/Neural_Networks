import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk

def load_model():
    """Load the trained model."""
    return tf.keras.models.load_model("best_model.h5")

def preprocess_image(image_path):
    """Load and preprocess an image for model prediction."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (30, 30))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image_path):
    """Predict the class of a given image."""
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def upload_image():
    """Open file dialog to select an image and display prediction."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = Image.open(file_path)
    image = image.resize((200, 200))
    img_display = ImageTk.PhotoImage(image)
    image_label.config(image=img_display)
    image_label.image = img_display
    
    predicted_class, confidence = predict(file_path)
    label_text.set(f"Prediction: {class_labels.get(predicted_class, 'Unknown')}\nConfidence: {confidence:.2f}")

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

# Load model
model = load_model()

# Create GUI window
root = tk.Tk()
root.title("Traffic Sign Classifier")
root.geometry("400x400")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

label_text = tk.StringVar()
label_text.set("Select an image to classify.")
prediction_label = tk.Label(root, textvariable=label_text, font=("Arial", 14))
prediction_label.pack(pady=10)

root.mainloop()
