"""Database operations for face recognition system."""

import sqlite3
import shutil
import os
from config import DB_FILE, DB_IMAGES_FOLDER


def create_database():
    """Create the SQLite database and faces table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def save_face_to_db(image_path, name):
    """Save a face image to the database.
    
    Args:
        image_path (str): Path to the source image
        name (str): Name of the person
        
    Returns:
        str: Path where the image was saved
    """
    # Create unique filename
    image_filename = f"{name}_{len(os.listdir(DB_IMAGES_FOLDER)) + 1}.jpg"
    image_save_path = os.path.join(DB_IMAGES_FOLDER, image_filename)
    
    # Copy image to database folder
    shutil.copy(image_path, image_save_path)
    
    # Save to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO faces (name, image_path) VALUES (?, ?)',
        (name, image_save_path)
    )
    conn.commit()
    conn.close()
    
    return image_save_path


def get_all_faces():
    """Retrieve all faces from the database.
    
    Returns:
        list: List of tuples (id, name, image_path, created_at)
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM faces ORDER BY created_at DESC')
    faces = cursor.fetchall()
    conn.close()
    return faces