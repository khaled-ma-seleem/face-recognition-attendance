"""Person recognition database operations."""

import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Union
from config import DB_FILE, DB_IMAGES_FOLDER


class PersonDatabaseManager:
    """Database manager for person recognition and attendance tracking."""

    def __init__(self, db_file: str = DB_FILE, images_folder: str = DB_IMAGES_FOLDER):
        """Initialize the database connection.

        Args:
            db_file (str): Path to SQLite database file
            images_folder (str): Path to folder for storing person images
        """
        self.db_file = db_file
        self.images_folder = images_folder
        self._create_tables()

    def _create_tables(self) -> None:
        """Create required database tables if they don't exist."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            # Persons table - using auto increment id as the only identifier
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'present',
                    FOREIGN KEY(person_id) REFERENCES persons(id),
                    UNIQUE(person_id, date)
                )
            ''')

            conn.commit()

    def save_cv_frame(self, frame: np.ndarray, filename: str) -> str:
        """Save OpenCV frame as image file.

        Args:
            frame (np.ndarray): OpenCV frame to save
            filename (str): Name for the saved file

        Returns:
            str: Full path where the image was saved
        """
        # Ensure images folder exists
        os.makedirs(self.images_folder, exist_ok=True)

        # Create full path
        file_path = os.path.join(self.images_folder, filename)

        # Save the frame
        cv2.imwrite(file_path, frame)

        return file_path

    # ========== PERSON MANAGEMENT ==========

    def add_person(self, image: np.ndarray, name: str) -> Tuple[int, str]:
        """Add a new person to the database with their image.

        Args:
            image (np.ndarray): OpenCV frame containing the person's face
            name (str): Person's name

        Returns:
            Tuple[int, str]: (person_id, image_path) where the image was saved
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            # Get the next ID (current max + 1)
            cursor.execute('SELECT COALESCE(MAX(id), 0) + 1 FROM persons')
            next_id = cursor.fetchone()[0]

            # Create filename: id_name.jpg
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"{next_id}_{name}_{timestamp}.jpg"

            # Save the image
            image_path = self.save_cv_frame(image, image_filename)

            # Insert into database
            cursor.execute('''
                INSERT INTO persons (name, image_path) VALUES (?, ?)
            ''', (name, image_path))

            person_id = cursor.lastrowid
            conn.commit()

            return person_id, image_path

    def get_all_persons(self) -> List[Tuple]:
        """Retrieve all persons from the database.

        Returns:
            List[Tuple]: List of person records (id, name, image_path, created_at)
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM persons ORDER BY created_at DESC')
            return cursor.fetchall()

    def find_person_by_id(self, person_id: int) -> Optional[Tuple]:
        """Find a person by their ID.

        Args:
            person_id (int): The person ID to search for

        Returns:
            Optional[Tuple]: Person record if found, None otherwise
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
            return cursor.fetchone()

    def find_persons_by_name(self, name: str) -> List[Tuple]:
        """Find all persons with a specific name.

        Args:
            name (str): The name to search for

        Returns:
            List[Tuple]: List of matching person records
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM persons
                WHERE name LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{name}%',))
            return cursor.fetchall()

    def extract_id_from_image_path(self, image_path: str) -> Optional[int]:
        """Extract person ID from image file path.

        Args:
            image_path (str): Path to the image file

        Returns:
            Optional[int]: Person ID extracted from filename or None if invalid format
        """
        try:
            # Extract filename from path
            filename = os.path.basename(image_path)

            # Split by underscore and get first part (ID)
            # Expected format: id_name_timestamp.jpg
            parts = filename.split("_")

            if len(parts) >= 1:
                return int(parts[0])  # First part is the ID

            return None

        except (ValueError, IndexError) as e:
            print(f"Error extracting ID from {image_path}: {e}")
            return None

    def remove_person(self, person_id: int) -> bool:
        """Remove a person from the database and delete their image file.

        Args:
            person_id (int): The database ID of the person to remove

        Returns:
            bool: True if removal was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # Get the image path before deletion
                cursor.execute('SELECT image_path FROM persons WHERE id = ?', (person_id,))
                result = cursor.fetchone()

                if result:
                    image_path = result[0]

                    # Delete from database (attendance records will be deleted due to foreign key)
                    cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
                    conn.commit()

                    # Delete the image file if it exists
                    if os.path.exists(image_path):
                        os.remove(image_path)

                    return True

                return False

        except Exception as e:
            print(f"Error removing person: {e}")
            return False

    # ========== ATTENDANCE MANAGEMENT ==========

    def record_attendance(self, person_id: int, day: Union[str, date] = None,
                         status: str = "present") -> bool:
        """Record attendance for a person for a specific day.

        Args:
            person_id (int): ID from the persons table
            day (Union[str, date]): Date for attendance (YYYY-MM-DD format or date object)
                                   If None, uses current date
            status (str): Attendance status (default "present")

        Returns:
            bool: True if attendance was recorded, False if already exists for that day
        """
        if day is None:
            day = date.today()
        elif isinstance(day, str):
            day = datetime.strptime(day, '%Y-%m-%d').date()

        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO attendance (person_id, date, status)
                    VALUES (?, ?, ?)
                ''', (person_id, day, status))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Attendance already exists for this person on this date
            return False
        except Exception as e:
            print(f"Error recording attendance: {e}")
            return False

    def get_all_attendance_records(self) -> List[Tuple]:
        """Retrieve all attendance records with person information.

        Returns:
            List[Tuple]: List of attendance records
                (attendance_id, person_id, name, date, timestamp, status)
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, a.person_id, p.name, a.date, a.timestamp, a.status
                FROM attendance a
                JOIN persons p ON a.person_id = p.id
                ORDER BY a.date DESC, a.timestamp DESC
            ''')
            return cursor.fetchall()

    def get_attendance_by_date(self, day: Union[str, date]) -> List[Tuple]:
        """Get attendance records for a specific date.

        Args:
            day (Union[str, date]): Date in YYYY-MM-DD format or date object

        Returns:
            List[Tuple]: List of attendance records for the specified date
        """
        if isinstance(day, str):
            day = datetime.strptime(day, '%Y-%m-%d').date()

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, a.person_id, p.name, a.date, a.timestamp, a.status
                FROM attendance a
                JOIN persons p ON a.person_id = p.id
                WHERE a.date = ?
                ORDER BY a.timestamp DESC
            ''', (day,))
            return cursor.fetchall()

    def get_attendance_by_person_id(self, person_id: int) -> List[Tuple]:
        """Get attendance records for a specific person.

        Args:
            person_id (int): Person ID to get attendance for

        Returns:
            List[Tuple]: List of attendance records for the person
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, a.person_id, p.name, a.date, a.timestamp, a.status
                FROM attendance a
                JOIN persons p ON a.person_id = p.id
                WHERE a.person_id = ?
                ORDER BY a.date DESC, a.timestamp DESC
            ''', (person_id,))
            return cursor.fetchall()

    # ========== DELETION OPERATIONS ==========

    def delete_attendance_table(self) -> bool:
        """Delete all records from the attendance table only.

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM attendance')
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting attendance records: {e}")
            return False

    def delete_persons_table(self) -> bool:
        """Delete all records from the persons table and remove all images.

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                # Delete attendance first due to foreign key constraints
                cursor.execute('DELETE FROM attendance')
                cursor.execute('DELETE FROM persons')
                conn.commit()

            # Remove all images
            if os.path.exists(self.images_folder):
                for filename in os.listdir(self.images_folder):
                    file_path = os.path.join(self.images_folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            return True
        except Exception as e:
            print(f"Error deleting persons table: {e}")
            return False

    def reset_database(self) -> bool:
        """Completely reset the database by deleting the file and recreating tables.

        Returns:
            bool: True if reset was successful, False otherwise
        """
        try:
            # Delete database file
            if os.path.exists(self.db_file):
                os.remove(self.db_file)

            # Delete all images
            if os.path.exists(self.images_folder):
                for filename in os.listdir(self.images_folder):
                    file_path = os.path.join(self.images_folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # Recreate tables
            self._create_tables()

            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
