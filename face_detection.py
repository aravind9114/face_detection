import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        print(f"Processing frame {self.frame_count}")

        img = frame.to_ndarray(format="bgr24")
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale")

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"Detected faces: {faces}")

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

st.title("Real-Time Face Detection")
st.write("Using OpenCV and Streamlit to detect faces in real-time.")

# Print debug information
print("Application started")

try:
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    print("Video streamer initialized")
except Exception as e:
    st.error(f"An error occurred: {e}")
    print(f"Error: {e}")

# Print debug information
print("Application ended")
