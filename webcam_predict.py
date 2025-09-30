import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load the trained ASL model
model = load_model("/Users/akhilaa/Desktop/asl_project/web_app/asl_model.h5")

# Mapping of class indices to letters
class_indices = {
    0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J",
    10:"K",11:"L",12:"M",13:"N",14:"O",15:"P",16:"Q",17:"R",18:"S",19:"T",
    20:"U",21:"V",22:"W",23:"X",24:"Y",25:"Z",26:"nothing",27:"space"
}

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Optional: flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI) for hand
    x_start, y_start, width, height = 100, 100, 300, 300
    roi = frame[y_start:y_start+height, x_start:x_start+width]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Make prediction
    pred = model.predict(roi_expanded)
    predicted_class = int(np.argmax(pred, axis=1)[0])
    label = class_indices[predicted_class]

    # Display prediction on the frame
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Draw rectangle for ROI
    cv2.rectangle(frame, (x_start, y_start), (x_start+width, y_start+height), (255, 0, 0), 2)

    # Show the webcam feed
    cv2.imshow("ASL Predictor", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
