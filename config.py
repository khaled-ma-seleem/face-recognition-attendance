"""Configuration settings for the attendance recording application."""
import os

# Database settings
DATA_FOLDER = 'data'
DB_FILE = f'{DATA_FOLDER}/faces.db'
DB_IMAGES_FOLDER = f'{DATA_FOLDER}/db_images'
TEMP_FACES_FOLDER = f'{DATA_FOLDER}/temp_faces'

# Face detection settings
RECOGNITION_THRESHOLD = 0.7
FACE_DETECTION_MIN_SIZE = (40, 40)
BOUNDING_BOX_OFFSET = 20

# Colors (BGR format for OpenCV)
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green
BOUNDING_BOX_THICKNESS = 4

# Ensure data directories exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DB_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TEMP_FACES_FOLDER, exist_ok=True)