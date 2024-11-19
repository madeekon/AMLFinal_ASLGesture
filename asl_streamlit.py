import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the fine-tuned model
try:
    model = load_model("asl_model_finetuned.keras")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Define class labels
classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
           11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
           21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space', 27: 'del', 28: 'nothing'}

# Function to preprocess the image/frame
def preprocess_frame(frame):
    img_resized = cv2.resize(frame, (224, 224))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

# Function to predict gesture
def predict_gesture(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame, verbose=0)
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return classes[predicted_label], confidence

# Streamlit UI
st.title("ASL Gesture Recognition")
st.write("Upload an image or use the webcam to predict ASL gestures.")

# File Upload Section
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    
    predicted_label, confidence = predict_gesture(image)
    st.write(f"**Predicted Gesture:** {predicted_label} ({confidence:.2f}%)")

# Webcam Section
st.write("Alternatively, you can use your webcam:")

start_webcam = st.button("START")
stop_webcam = st.button("STOP")

if start_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        predicted_label, confidence = predict_gesture(frame)
        
        label_text = f"{predicted_label} ({confidence:.2f}%)"
        cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", caption="Webcam Feed")
        
        # Break the loop when STOP is clicked
        if stop_webcam:
            break

    cap.release()
    stframe.empty()
    st.success("Webcam stopped.")



# cd "C:\Users\Admin\Desktop\sabaq(C3T1)\C3T1\Applied ML\aml_final"
# streamlit run asl_streamlit.py