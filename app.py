"""Real-time face detection and recognition application using Streamlit."""

import streamlit as st
import numpy as np
import cv2
import os
import uuid
from datetime import datetime

from src.database import create_database, save_face_to_db
from src.face_utils import detect_faces, draw_bounding_boxes, crop_face_with_offset, recognize_face
from config import CAMERA_INDEX


def initialize_app():
    """Initialize the application and database."""
    create_database()
    if "run_camera" not in st.session_state:
        st.session_state["run_camera"] = False


def process_frame_and_recognize(frame):
    """Process frame for face detection and recognition.
    
    Args:
        frame: OpenCV frame from camera
        
    Returns:
        frame: Processed frame with bounding boxes
    """
    faces = detect_faces(frame)
    
    if len(faces) > 0:
        # Process first detected face
        face_coords = faces[0]
        cropped_face = crop_face_with_offset(frame, face_coords)
        
        # Save temporary image for recognition
        temp_path = f"temp_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_path, cropped_face)
        
        try:
            # Attempt recognition
            results = recognize_face(temp_path)
            
            if len(results) == 0:
                # Unknown face - save to database
                random_id = str(uuid.uuid4())[:6]
                name = f"UNKNOWN_{random_id}"
                save_face_to_db(temp_path, name)
                st.info(f"New face saved as: {name}")
            else:
                # Known face found
                st.success("Face recognized!")
                st.dataframe(results[['identity', 'distance']], use_container_width=True)
                
        except Exception as e:
            st.error(f"Recognition error: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Draw bounding boxes
    return draw_bounding_boxes(frame, faces)


def run_camera():
    """Main camera loop using Streamlit camera input (for Codespaces)."""
    frame_placeholder = st.empty()

    try:
        while st.session_state["run_camera"]:
            img_file_buffer = st.camera_input("Capture a picture")

            if img_file_buffer is None:
                st.warning("No image captured yet.")
                break

            # Convert uploaded frame to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            np_array = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Process frame
            processed_frame = process_frame_and_recognize(frame)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb,
                caption="Captured Frame",
                use_container_width=True
            )

            # Break after one capture (camera_input is not continuous)
            break

    finally:
        st.success("Camera session ended.")


def main():
    """Main application interface."""
    st.set_page_config(
        page_title="Face Recognition App",
        page_icon="👤",
        layout="centered"
    )
    
    st.title("🎯 Real-time Face Recognition")
    st.markdown("*Detect and recognize faces using your camera*")
    
    initialize_app()
    
    # Camera controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_active = st.checkbox(
            "📹 Start Camera",
            value=st.session_state["run_camera"],
            help="Toggle camera on/off"
        )
        st.session_state["run_camera"] = camera_active
    
    with col2:
        if st.button("🗂️ View Database", help="Show all stored faces"):
            from src.database import get_all_faces
            faces = get_all_faces()
            if faces:
                st.write(f"**Total faces in database: {len(faces)}**")
                for face in faces[:5]:  # Show latest 5
                    st.write(f"• {face[1]} - {face[3]}")
            else:
                st.write("No faces in database yet.")
    
    # Status information
    if camera_active:
        st.info("📸 Use the camera widget below to capture a picture.")
        run_camera()
    else:
        st.info("📷 Click 'Start Camera' to begin face detection.")


if __name__ == "__main__":
    main()