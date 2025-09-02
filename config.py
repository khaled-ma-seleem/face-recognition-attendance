"""Configuration settings for the face detection application."""

import os

# Database settings
DB_FILE = 'data/faces.db'
DB_IMAGES_FOLDER = 'data/db_images'

# Face detection settings
RECOGNITION_THRESHOLD = 0.6
FACE_DETECTION_MIN_SIZE = (40, 40)
BOUNDING_BOX_OFFSET = 20

# Camera settings
CAMERA_INDEX = 0

# Colors (BGR format for OpenCV)
BOUNDING_BOX_COLOR = (0, 255, 0)
BOUNDING_BOX_THICKNESS = 4

# Ensure data directories exist
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
os.makedirs(DB_IMAGES_FOLDER, exist_ok=True)