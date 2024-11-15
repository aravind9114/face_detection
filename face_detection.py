import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

st.title("Real-Time Face Detection")
st.write("Using OpenCV and Streamlit to detect faces in real-time.")

# Create session state for controlling the video stream
if 'run' not in st.session_state:
    st.session_state.run = False

# Start button
if st.button("Start"):
    st.session_state.run = True

# Stop button
if st.button("Stop"):
    st.session_state.run = False

# Ensure webcam access and handle errors
if st.session_state.run:
    try:
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.run = False
else:
    st.write("Click 'Start' to begin the video stream.")

# Clear video placeholder and release resources when stopped
if not st.session_state.run:
    st.write("Video capture stopped.")
