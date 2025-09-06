import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime, date
import os

# Import your backend modules
from config import DB_IMAGES_FOLDER, TEMP_FACES_FOLDER
from src.database import PersonDatabaseManager
from src.face_utils import FaceRecognitionProcessor

# Initialize session state
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'face_data' not in st.session_state:
    st.session_state.face_data = []  # List of face info dictionaries
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'db_update_trigger' not in st.session_state:
    st.session_state.db_update_trigger = 0

# Initialize backend components
@st.cache_resource
def initialize_components():
    """Initialize database manager and face processor."""
    db_manager = PersonDatabaseManager()
    face_processor = FaceRecognitionProcessor()
    return db_manager, face_processor

def reset_session_state():
    """Reset session state for new capture."""
    st.session_state.captured_image = None
    st.session_state.face_data = []
    st.session_state.processing_complete = False

def trigger_db_update():
    """Trigger database update in sidebar."""
    st.session_state.db_update_trigger += 1

def camera_section():
    """Handle camera control and image capture."""
    st.header("📸 Camera Control")
    
    # Camera enable/disable checkbox
    camera_enabled = st.checkbox("Enable Camera", value=st.session_state.camera_enabled)
    st.session_state.camera_enabled = camera_enabled
    
    if camera_enabled:
        # Camera input
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            # Convert to OpenCV format
            image_bytes = camera_input.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.captured_image = opencv_image
            
            return True
    else:
        st.info("Enable camera checkbox to access camera")
    
    return False

def detect_and_recognize_section(face_processor, db_manager):
    """Single button to detect and recognize faces."""
    if st.session_state.captured_image is not None and not st.session_state.processing_complete:
        st.header("🔍 Face Detection & Recognition")
        
        if st.button("Detect & Recognize Faces", type="primary", use_container_width=True):
            with st.spinner("Processing faces..."):
                # Detect faces
                faces = face_processor.detect_faces(st.session_state.captured_image)
                
                if faces:
                    st.toast(f"Detected {len(faces)} face(s)")
                    
                    # Crop faces
                    cropped_faces = face_processor.crop_all_faces(
                        st.session_state.captured_image, faces
                    )
                    
                    # Recognize faces
                    recognition_results = face_processor.recognize_multiple_faces(cropped_faces)
                    person_ids = face_processor.get_recognized_person_ids(recognition_results)
                    
                    # Process each face and create face data
                    face_data = []
                    for i, (face_coords, cropped_face, result, person_id) in enumerate(
                        zip(faces, cropped_faces, recognition_results, person_ids)
                    ):
                        
                        face_info = {
                            'index': i,
                            'coords': face_coords,
                            'cropped_face': cropped_face,
                            'recognition_result': result,
                            'person_id': person_id,
                            'status': 'known' if person_id else 'unknown',
                            'person_details': None,
                            'attendance_recorded': False
                        }
                        
                        # Get person details if known
                        if person_id:
                            person_details = db_manager.find_person_by_id(person_id)
                            face_info['person_details'] = person_details
                            
                            # Check if attendance already recorded today
                            today_attendance = db_manager.get_attendance_by_date(date.today())
                            face_info['attendance_recorded'] = any(
                                record[1] == person_id for record in today_attendance
                            )
                        
                        face_data.append(face_info)
                    
                    st.session_state.face_data = face_data
                    st.session_state.processing_complete = True
                    
                    return True
                else:
                    st.warning("No faces detected in the image")
                    return False
    
    return False

def display_face_cards(db_manager):
    """Display individual cards for each detected face, handling dismissed ones."""
    if st.session_state.face_data:
        st.header("👥 Detected Faces")
        
        # Filter out dismissed faces
        active_faces = [face for i, face in enumerate(st.session_state.face_data) 
                       if not face.get('dismissed', False)]
        
        for i, face_info in enumerate(active_faces):
            # Find the original index in session state
            original_idx = next((idx for idx, f in enumerate(st.session_state.face_data) 
                               if not f.get('dismissed', False) and f == face_info), 0)
            
            with st.container():
                st.markdown(f"### Face {i + 1}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display cropped face
                    face_rgb = cv2.cvtColor(face_info['cropped_face'], cv2.COLOR_BGR2RGB)
                    st.image(face_rgb, caption=f"Face {i + 1}", width=200)
                
                with col2:
                    if face_info['status'] == 'known':
                        display_known_person_card(face_info, db_manager, original_idx)
                    elif face_info['status'] == 'unknown':
                        display_unknown_person_card(face_info, db_manager, original_idx)
                    elif face_info['status'] == 'newly_added':
                        display_newly_added_person_card(face_info, db_manager, original_idx)
                
                st.markdown("---")
                
        # Show message if all faces were dismissed
        if not active_faces and st.session_state.face_data:
            st.info("All unknown faces have been dismissed.")

def display_known_person_card(face_info, db_manager, face_index):
    """Display card for known person."""
    person_details = face_info['person_details']
    person_id, name, image_path, created_at = person_details
    
    # Person information
    st.success(f"**Known Person: {name}**")
    st.info(f"**Person ID:** {person_id}")
    st.info(f"**Added on:** {created_at}")
    
    # Confidence score
    best_match = FaceRecognitionProcessor().get_best_match(face_info['recognition_result'])
    if best_match:
        st.info(f"**Recognition Confidence:** {best_match[1]:.2f}")
    
    # Attendance status and button
    if face_info['attendance_recorded']:
        st.success("✅ **Attendance recorded today**")
    else:
        if st.button(f"📋 Record Attendance", key=f"attend_{face_index}", type="secondary"):
            with st.spinner("Recording attendance..."):
                success = db_manager.record_attendance(person_id)
                if success:
                    st.success(f"✅ Attendance recorded for {name}")
                    # Update face info
                    st.session_state.face_data[face_index]['attendance_recorded'] = True
                    trigger_db_update()
                    st.rerun()
                else:
                    st.warning("⚠️ Attendance already recorded today")

def display_unknown_person_card(face_info, db_manager, face_index):
    """Display card for unknown person with dismiss option."""
    # Create a container for the card
    card_container = st.container()
    
    with card_container:
        st.warning("**Unknown Person**")
        
        # Name input
        name = st.text_input(
            f"Enter name for Face {face_index + 1}:",
            key=f"name_input_{face_index}",
            placeholder="Enter person's name..."
        )
        
        # Buttons row: Add and Dismiss side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button(f"➕ Add Person", key=f"add_{face_index}", type="primary"):
                if name.strip():
                    with st.spinner("Adding person to database..."):
                        try:
                            person_id, image_path = db_manager.add_person(
                                face_info['cropped_face'], name.strip()
                            )
                            
                            # Update face info
                            person_details = db_manager.find_person_by_id(person_id)
                            st.session_state.face_data[face_index]['person_id'] = person_id
                            st.session_state.face_data[face_index]['person_details'] = person_details
                            st.session_state.face_data[face_index]['status'] = 'newly_added'
                            st.session_state.face_data[face_index]['attendance_recorded'] = False
                            st.session_state.face_data[face_index]['dismissed'] = False
                            
                            trigger_db_update()
                            st.success(f"✅ Added {name} to database (ID: {person_id})")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to add {name}: {str(e)}")
        
        with col2:
            if st.button("❌ Dismiss", key=f"dismiss_{face_index}", type="secondary"):
                # Mark this face as dismissed
                st.session_state.face_data[face_index]['dismissed'] = True
                st.rerun()

def display_newly_added_person_card(face_info, db_manager, face_index):
    """Display card for newly added person."""
    person_details = face_info['person_details']
    person_id, name, image_path, created_at = person_details
    
    # Person information with success styling
    st.success(f"**✅ Successfully Added: {name}**")
    st.info(f"**Person ID:** {person_id}")
    st.info(f"**Added on:** {created_at}")
    
    # Attendance button
    if face_info['attendance_recorded']:
        st.success("✅ **Attendance recorded today**")
    else:
        if st.button(f"📋 Record Attendance", key=f"attend_new_{face_index}", type="secondary"):
            with st.spinner("Recording attendance..."):
                success = db_manager.record_attendance(person_id)
                if success:
                    st.success(f"✅ Attendance recorded for {name}")
                    # Update face info
                    st.session_state.face_data[face_index]['attendance_recorded'] = True
                    trigger_db_update()
                    st.rerun()
                else:
                    st.warning("⚠️ Attendance already recorded today")

def sidebar_database_info(db_manager):
    """Enhanced sidebar with database information and dropdowns."""
    with st.sidebar:
        st.header("📊 Database Information")
        
        # Trigger update based on session state
        _ = st.session_state.db_update_trigger
        
        # Get current statistics
        all_persons = db_manager.get_all_persons()
        today_attendance = db_manager.get_attendance_by_date(date.today())
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Persons", len(all_persons))
        with col2:
            st.metric("Today's Attendance", len(today_attendance))
        
        st.markdown("---")
        
        # Persons in Database Dropdown
        st.subheader("👥 Persons in Database")
        if all_persons:
            with st.expander(f"View All Persons ({len(all_persons)})", expanded=False):
                for person in all_persons:
                    person_id, name, image_path, created_at = person
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{name}** (ID: {person_id})")
                        st.caption(f"Added: {created_at}")
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_person_{person_id}", 
                                   help="Delete person", type="secondary"):
                            with st.spinner(f"Deleting {name}..."):
                                success = db_manager.remove_person(person_id)
                                if success:
                                    st.toast(f"✅ Deleted {name}")
                                    trigger_db_update()
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete {name}")
                    
                    st.markdown("---")
        else:
            st.info("No persons in database")
        
        # Today's Attendance Dropdown
        st.subheader("📅 Today's Attendance")
        if today_attendance:
            with st.expander(f"View Today's Records ({len(today_attendance)})", expanded=False):
                for record in today_attendance:
                    attendance_id, person_id, name, record_date, timestamp, status = record
                    st.write(f"**{name}** (ID: {person_id})")
                    st.caption(f"Time: {timestamp}")
                    st.caption(f"Status: {status}")
                    st.markdown("---")
        else:
            st.info("No attendance recorded today")

def control_panel():
    """Control panel with various options."""
    with st.sidebar:
        st.markdown("---")
        st.header("🎛️ Control Panel")
        
        if st.button("🔄 Reset/New Capture", use_container_width=True):
            reset_session_state()
            st.rerun()
        
        st.markdown("---")
        
        # Database management
        st.subheader("🗑️ Database Management")
        
        with st.expander("Danger Zone"):
            st.warning("⚠️ These actions cannot be undone!")
            
            if st.button("Clear All Attendance", type="secondary"):
                db_manager, _ = initialize_components()
                if db_manager.delete_attendance_table():
                    st.success("All attendance records cleared!")
                    trigger_db_update()
                    st.rerun()
            
            if st.button("Clear All Persons", type="secondary"):
                db_manager, _ = initialize_components()
                if db_manager.delete_persons_table():
                    st.success("All persons cleared!")
                    trigger_db_update()
                    st.rerun()
            
            if st.button("Reset Entire Database", type="secondary"):
                db_manager, _ = initialize_components()
                if db_manager.reset_database():
                    st.success("Database reset successfully!")
                    trigger_db_update()
                    reset_session_state()
                    st.rerun()

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Face Recognition Attendance System",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Face Recognition Attendance System")
    st.markdown("---")
    
    # Initialize components
    db_manager, face_processor = initialize_components()
    
    # Sidebar components
    sidebar_database_info(db_manager)
    control_panel()
    
    # Main workflow
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Step 1: Camera Control
        image_captured = camera_section()
        
        # Step 2: Detect and Recognize (Single Button)
        if image_captured:
            detect_and_recognize_section(face_processor, db_manager)
        
        # Step 3: Display Face Cards
        if st.session_state.processing_complete:
            display_face_cards(db_manager)
    
    with col2:
        st.markdown("### 📝 How to Use")
        st.markdown("""
        1. ☑️ **Enable Camera**: Check the camera checkbox
        2. 📸 **Take Picture**: Capture image with camera
        3. 🔍 **Process**: Click "Detect & Recognize Faces"
        4. 👤 **Handle Faces**: 
           - **Known**: Record attendance
           - **Unknown**: Add name and person
           - **New**: Record attendance after adding
        5. 📊 **View Data**: Check sidebar for database info
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - ✅ Good lighting improves detection
        - 🎯 Face camera directly
        - 👥 Multiple people can be processed
        - 📱 Sidebar updates automatically
        - 🔄 Use reset for new capture
        """)
        
        # Status indicator
        if st.session_state.camera_enabled:
            st.success("📸 Camera Enabled")
        else:
            st.info("📸 Camera Disabled")

if __name__ == "__main__":
    main()