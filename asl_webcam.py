import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the fine-tuned model
model = load_model("asl_model_finetuned.keras")  # Ensure the file name matches your saved fine-tuned model
print("Fine-tuned model loaded successfully!")

# Map indices back to class names (ensure this matches your training classes)
index_to_class = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                  8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                  15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                  22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

# Webcam configuration
IMG_SIZE = 64  # Match the input size of the model
font = cv2.FONT_HERSHEY_SIMPLEX

# Start webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the webcam feed.")

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the original webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    # Resize to the expected input size (224x224)
    resized_frame = cv2.resize(frame, (224, 224))
    reshaped_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    reshaped_frame = reshaped_frame / 255.0  # Normalize the image to [0, 1]

    # Predict using the fine-tuned model
    prediction = model.predict(reshaped_frame, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_class[predicted_index]

    # Display prediction on the webcam feed
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("ASL Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# cd "C:\Users\Admin\Desktop\sabaq(C3T1)\C3T1\Applied ML\aml_final"
# python asl_webcam.py