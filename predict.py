import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Path to new image
img_path = "/Users/akhilaa/Desktop/asl_project/backend/model/test_image.jpg"
  # replace with your image path

# Preprocess the image (resize to match training size, e.g. 64x64)
img = image.load_img(img_path, target_size=(64, 64))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
img_array = img_array / 255.0  # normalize like training

# Make prediction
pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)[0]

# Load class labels (assuming 29 classes, use your generator’s class_indices)
class_indices = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
    25: "Z", 26: "del", 27: "nothing", 28: "space"
}

print("Predicted class:", class_indices[predicted_class])
