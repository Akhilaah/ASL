import os
import sys
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

print("Script started", flush=True)

try:
    model = load_model("asl_model.h5")
    print("Model loaded successfully", flush=True)
except Exception as e:
    print("Error loading model:", e, flush=True)
    sys.exit()

test_folder = "test_images"
files = os.listdir(test_folder)
print("Files found in folder:", files, flush=True)

class_indices = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
    25: "Z", 26: "del", 27: "nothing", 28: "space"
}

for img_name in files:
    print("Processing:", img_name, flush=True)
    img_path = os.path.join(test_folder, img_name)

    try:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred, axis=1)[0]
        print(f"{img_name} → {class_indices[predicted_class]}", flush=True)

    except Exception as e:
        print(f"Error processing {img_name}: {e}", flush=True)

print("Script ended", flush=True)
