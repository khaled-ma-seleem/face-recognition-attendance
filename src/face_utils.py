"""Face detection and recognition utilities."""

import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from typing import List, Tuple, Optional, Union
from config import (
    RECOGNITION_THRESHOLD,
    FACE_DETECTION_MIN_SIZE,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    BOUNDING_BOX_OFFSET,
    DB_IMAGES_FOLDER,
    TEMP_FACES_FOLDER
)


class FaceRecognitionProcessor:
    """Face detection and recognition processor using OpenCV and DeepFace."""

    def __init__(self, db_images_folder: str = DB_IMAGES_FOLDER,
                 recognition_threshold: float = RECOGNITION_THRESHOLD):
        """Initialize the face processor.

        Args:
            db_images_folder (str): Path to folder containing face database images
            recognition_threshold (float): Threshold for face recognition matching
        """
        self.db_images_folder = db_images_folder
        self.recognition_threshold = recognition_threshold
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def convert_bytes_to_cv_frame(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV frame.

        Args:
            image_bytes (bytes): Image data as bytes

        Returns:
            np.ndarray: OpenCV frame
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def load_image_as_cv_frame(self, image_path: str) -> np.ndarray:
        """Load image from path as OpenCV frame.

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: OpenCV frame
        """
        return cv2.imread(image_path)

    def _process_image_input(self, image_input: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """Process different types of image input and return OpenCV frame.

        Args:
            image_input: Can be path (str), OpenCV frame (np.ndarray), or image bytes

        Returns:
            np.ndarray: OpenCV frame

        Raises:
            ValueError: If image input type is not supported
        """
        if isinstance(image_input, str):
            # Image path
            return self.load_image_as_cv_frame(image_input)
        elif isinstance(image_input, np.ndarray):
            # OpenCV frame
            return image_input
        elif isinstance(image_input, bytes):
            # Image bytes
            return self.convert_bytes_to_cv_frame(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # ========== FACE DETECTION ==========

    def detect_faces(self, image_input: Union[str, np.ndarray, bytes]) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image using Haar cascades.

        Args:
            image_input: Can be image path (str), OpenCV frame (np.ndarray), or image bytes

        Returns:
            List[Tuple[int, int, int, int]]: List of face bounding boxes (x, y, w, h)
        """
        frame = self._process_image_input(image_input)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=FACE_DETECTION_MIN_SIZE
        )

        return faces.tolist() if len(faces) > 0 else []

    # ========== FACE CROPPING ==========

    def crop_face_with_padding(self, image_input: Union[str, np.ndarray, bytes],
                              face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop face from image with padding offset.

        Args:
            image_input: Can be image path (str), OpenCV frame (np.ndarray), or image bytes
            face_coords (Tuple[int, int, int, int]): Face coordinates (x, y, w, h)

        Returns:
            np.ndarray: Cropped face image with padding
        """
        frame = self._process_image_input(image_input)
        x, y, w, h = face_coords

        # Apply padding with boundary checks
        x_start = max(x - BOUNDING_BOX_OFFSET, 0)
        y_start = max(y - BOUNDING_BOX_OFFSET, 0)
        x_end = min(x + w + BOUNDING_BOX_OFFSET, frame.shape[1])
        y_end = min(y + h + BOUNDING_BOX_OFFSET, frame.shape[0])

        return frame[y_start:y_end, x_start:x_end]

    def crop_all_faces(self, image_input: Union[str, np.ndarray, bytes],
                      faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Crop all detected faces from image.

        Args:
            image_input: Can be image path (str), OpenCV frame (np.ndarray), or image bytes
            faces (List[Tuple[int, int, int, int]]): List of face bounding boxes

        Returns:
            List[np.ndarray]: List of cropped face images
        """
        cropped_faces = []
        for face_coords in faces:
            cropped_face = self.crop_face_with_padding(image_input, face_coords)
            cropped_faces.append(cropped_face)
        return cropped_faces

    # ========== FACE RECOGNITION ==========

    def save_temp_face(self, face_image: np.ndarray, face_index: int) -> str:
        """Save a cropped face temporarily for recognition.

        Args:
            face_image (np.ndarray): Cropped face image
            face_index (int): Index of the face

        Returns:
            str: Path to the saved temporary image
        """
        os.makedirs(TEMP_FACES_FOLDER, exist_ok=True)
        temp_path = os.path.join(TEMP_FACES_FOLDER, f"temp_face_{face_index}.jpg")
        cv2.imwrite(temp_path, face_image)
        return temp_path

    def recognize_single_face(self, image_input: Union[str, np.ndarray, bytes]) -> pd.DataFrame:
        """Recognize a single face using DeepFace by comparing with database.

        Args:
            image_input: Can be image path (str), OpenCV frame (np.ndarray), or image bytes

        Returns:
            pd.DataFrame: Recognition results with similarity scores
        """
        try:
            # If input is not a path, save it temporarily
            if isinstance(image_input, str):
                image_path = image_input
                temp_file = False
            else:
                frame = self._process_image_input(image_input)
                image_path = self.save_temp_face(frame, 0)
                temp_file = True

            results = DeepFace.find(
                img_path=image_path,
                db_path=self.db_images_folder,
                enforce_detection=False,
                threshold=self.recognition_threshold,
                silent=True
            )

            # Clean up temporary file if created
            if temp_file and os.path.exists(image_path):
                os.remove(image_path)

            if results and len(results) > 0:
                return pd.concat(results, ignore_index=True)
            return pd.DataFrame()

        except Exception as e:
            print(f"Recognition error: {e}")
            return pd.DataFrame()

    def recognize_multiple_faces(self, face_images: List[np.ndarray]) -> List[pd.DataFrame]:
        """Recognize multiple faces in batch processing.

        Args:
            face_images (List[np.ndarray]): List of cropped face images

        Returns:
            List[pd.DataFrame]: List of recognition results for each face
        """
        results = []
        for i, face_image in enumerate(face_images):
            result = self.recognize_single_face(face_image)
            results.append(result)
        return results

    def get_best_match(self, recognition_result: pd.DataFrame) -> Optional[Tuple[str, float]]:
        """Get the best match from recognition results.

        Args:
            recognition_result (pd.DataFrame): DataFrame from face recognition

        Returns:
            Optional[Tuple[str, float]]: Best match (identity_path, confidence) or None
        """
        if recognition_result.empty:
            return None

        # Get the top match (highest similarity/lowest distance)
        best_match = recognition_result.iloc[0]
        identity_path = best_match["identity"]

        # DeepFace returns distance (lower is better), convert to confidence
        distance = best_match.get("distance", 1.0)
        confidence = max(0.0, 1.0 - distance)  # Convert to confidence (0-1)

        return identity_path, confidence

    def extract_person_info_from_path(self, image_path: str) -> Optional[Tuple[int, str]]:
        """Extract person ID and name from image file path.

        Args:
            image_path (str): Path to the recognized image file

        Returns:
            Optional[Tuple[int, str]]: (person_id, name) or None if invalid format
        """
        try:
            # Extract filename from path
            filename = os.path.basename(image_path)

            # Split by underscore
            # Expected format: id_name_timestamp.jpg
            parts = filename.split("_")

            if len(parts) >= 2:
                person_id = int(parts[0])  # First part is ID
                name = parts[1]  # Second part is name
                return person_id, name

            return None

        except (ValueError, IndexError) as e:
            print(f"Error extracting person info from {image_path}: {e}")
            return None

    # ========== DRAWING UTILITIES ==========

    def draw_face_recognition_results(self, image_input: Union[str, np.ndarray, bytes],
                                    faces: List[Tuple[int, int, int, int]],
                                    recognition_results: List[pd.DataFrame]) -> np.ndarray:
        """Draw face recognition results with bounding boxes and labels.

        Args:
            image_input: Can be image path (str), OpenCV frame (np.ndarray), or image bytes
            faces (List[Tuple[int, int, int, int]]): List of face bounding boxes
            recognition_results (List[pd.DataFrame]): Recognition results for each face

        Returns:
            np.ndarray: Image with recognition results drawn
        """
        frame = self._process_image_input(image_input).copy()

        for i, (face_coords, result) in enumerate(zip(faces, recognition_results)):
            x, y, w, h = face_coords

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS)

            # Determine label and confidence based on recognition result
            if result.empty:
                # Unknown person
                name_label = f"Unknown {i+1}"
                confidence_label = "N/A"
                color = (0, 0, 255)  # Red for unknown
            else:
                best_match = self.get_best_match(result)
                if best_match:
                    person_info = self.extract_person_info_from_path(best_match[0])
                    if person_info:
                        person_id, name = person_info
                        name_label = f"{name} (ID: {person_id})"
                        confidence_label = f"Conf: {best_match[1]:.2f}"
                        color = (0, 255, 0)  # Green for recognized
                    else:
                        name_label = f"Unknown {i+1}"
                        confidence_label = "N/A"
                        color = (0, 0, 255)  # Red for unknown
                else:
                    name_label = f"Unknown {i+1}"
                    confidence_label = "N/A"
                    color = (0, 0, 255)  # Red for unknown

            # Calculate label background size
            name_size = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            conf_size = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            label_width = max(name_size[0], conf_size[0])
            label_height = name_size[1] + conf_size[1] + 15

            # Draw label background
            cv2.rectangle(frame, (x, y - label_height - 5),
                         (x + label_width + 10, y), color, -1)

            # Draw name label
            cv2.putText(frame, name_label, (x + 5, y - conf_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw confidence label
            cv2.putText(frame, confidence_label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def get_recognized_person_ids(self, recognition_results: List[pd.DataFrame]) -> List[Optional[int]]:
        """Extract person IDs from recognition results.

        Args:
            recognition_results (List[pd.DataFrame]): Recognition results for faces

        Returns:
            List[Optional[int]]: List of person IDs (None for unrecognized faces)
        """
        person_ids = []

        for result in recognition_results:
            if result.empty:
                person_ids.append(None)
            else:
                best_match = self.get_best_match(result)
                if best_match:
                    person_info = self.extract_person_info_from_path(best_match[0])
                    if person_info:
                        person_ids.append(person_info[0])  # person_id
                    else:
                        person_ids.append(None)
                else:
                    person_ids.append(None)

        return person_ids
