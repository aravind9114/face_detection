import cv2
import streamlit as st
import numpy as np

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up Streamlit app layout
st.title("Real-Time Face Detection")
st.write("Using OpenCV and Streamlit to detect faces in real-time.")

# Create session state for controlling the video stream
if 'run' not in st.session_state:
    st.session_state.run = False

# Start button
if st.button("Start", key="start_button"):
    st.session_state.run = True
    st.write("Starting video stream...")

# Stop button
if st.button("Stop", key="stop_button"):
    st.session_state.run = False
    st.write("Stopping video stream...")

# Create a placeholder for the video stream
video_placeholder = st.empty()

# Open the video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Start video stream when the run flag is set
if st.session_state.run:
    while st.session_state.run:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not read from video source.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detected faces
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

# Release resources when the stream is stopped
if not st.session_state.run:
    cap.release()
    video_placeholder.empty()  # Clear the video placeholder
    st.write("Video capture stopped.")
