import cv2
import streamlit as st
import numpy as np

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up Streamlit app layout
st.title("Real-Time Face Detection")
st.write("Using OpenCV and Streamlit to detect faces in real-time.")

# Enable camera input
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

# Process the captured image
if picture:
    # Convert the picture to an OpenCV image
    image = np.array(bytearray(picture.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert the image from BGR to RGB for display in Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the processed image
    st.image(image_rgb, channels="RGB")