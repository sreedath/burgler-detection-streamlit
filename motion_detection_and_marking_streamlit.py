import cv2
import streamlit as st
import numpy as np

st.title("Motion Detection and Marking")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
frame_placeholder = st.empty()
save_motion = st.checkbox("Save frames with motion", value=False)

if video_file is not None:
    tfile = open("temp_video.mp4", "wb")
    tfile.write(video_file.read())
    tfile.close()
    cap = cv2.VideoCapture("temp_video.mp4")
    ret, prev_frame = cap.read()
    if not ret:
        st.error("Could not read video.")
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0

        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                break

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, current_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < 20:
                    continue
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if motion_detected and save_motion and frame_count % 10 == 0:
                cv2.imwrite(f"motion_{frame_count}.jpg", current_frame)

            # Convert BGR to RGB for Streamlit display
            frame_placeholder.image(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            prev_gray = current_gray
            frame_count += 1

        cap.release()
