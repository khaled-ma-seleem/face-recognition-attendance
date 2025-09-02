"""Face detection and recognition utilities."""

import cv2
import pandas as pd
from deepface import DeepFace
from config import (
    RECOGNITION_THRESHOLD, 
    FACE_DETECTION_MIN_SIZE,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    BOUNDING_BOX_OFFSET,
    DB_IMAGES_FOLDER
)


def detect_faces(frame):
    """Detect faces in a frame using Haar cascades.
    
    Args:
        frame: OpenCV frame/image
        
    Returns:
        list: List of face bounding boxes (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=FACE_DETECTION_MIN_SIZE
    )
    
    return faces


def draw_bounding_boxes(frame, faces):
    """Draw bounding boxes around detected faces.
    
    Args:
        frame: OpenCV frame
        faces: List of face bounding boxes
        
    Returns:
        frame: Frame with bounding boxes drawn
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, 
            (x, y), 
            (x + w, y + h), 
            BOUNDING_BOX_COLOR, 
            BOUNDING_BOX_THICKNESS
        )
    return frame


def crop_face_with_offset(frame, face_coords):
    """Crop face from frame with padding offset.
    
    Args:
        frame: OpenCV frame
        face_coords: Tuple (x, y, w, h) of face coordinates
        
    Returns:
        cropped_face: Cropped face image
    """
    x, y, w, h = face_coords
    
    # Apply offset with boundary checks
    x_start = max(x - BOUNDING_BOX_OFFSET, 0)
    y_start = max(y - BOUNDING_BOX_OFFSET, 0)
    x_end = min(x + w + BOUNDING_BOX_OFFSET, frame.shape[1])
    y_end = min(y + h + BOUNDING_BOX_OFFSET, frame.shape[0])
    
    return frame[y_start:y_end, x_start:x_end]


def recognize_face(image_path):
    """Recognize face using DeepFace by comparing with database.
    
    Args:
        image_path (str): Path to the image to recognize
        
    Returns:
        pd.DataFrame: Recognition results
    """
    try:
        results = DeepFace.find(
            img_path=image_path,
            db_path=DB_IMAGES_FOLDER,
            enforce_detection=False,
            threshold=RECOGNITION_THRESHOLD,
            silent=True
        )
        
        if results and len(results) > 0:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return pd.DataFrame()