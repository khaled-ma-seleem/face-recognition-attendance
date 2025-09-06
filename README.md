# Face Recognition Attendance System

An attendance recoding system built with Streamlit, OpenCV, and DeepFace. The system allows users to capture images via webcam, detect and recognize faces, and record attendance for known individuals.

## Features

- **Face Detection**: Uses OpenCV Haar cascades for accurate face detection
- **Face Recognition**: Leverages DeepFace for robust face recognition with confidence scoring
- **Attendance Tracking**: Attendance recording with duplicate prevention
- **Person Management**: Add new persons to the database with face images
- **Web Interface**: Clean, intuitive Streamlit interface with live camera feed
- **Database Management**: SQLite database for storing person data and attendance records
- **Multi-face Processing**: Handle multiple faces in a single image
- **Visual Feedback**: Face detection with recognition results and confidence scores

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam/camera device
- Linux/Ubuntu (recommended for system dependencies)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies using Make:**
   ```bash
   make setup
   ```
   This will install Python dependencies and required system libraries.

3. **Run the application:**
   ```bash
   make run
   ```

### Alternative Installation

If Make is not available:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1

# Run the application
streamlit run app.py
```

## How to Use

### Basic Workflow

1. **Enable Camera**: Check the "Enable Camera" checkbox
2. **Capture Image**: Take a picture using the camera interface
3. **Process Faces**: Click "Detect & Recognize Faces" to analyze the image
4. **Handle Results**:
   - **Known persons**: Record attendance with one click
   - **Unknown persons**: Add name and register in database
   - **Newly added**: Record attendance after registration

### Managing the Database

The sidebar provides comprehensive database management:

- **View Statistics**: See total persons and today's attendance
- **Person Management**: View all registered persons with delete options
- **Attendance Records**: View today's attendance with timestamps
- **Database Operations**: Reset or clear specific data tables

### Advanced Features

- **Batch Processing**: Process multiple faces simultaneously
- **Clean Attendance**: Automatic duplicate attendance prevention
- **Face Dismissal**: Dismiss unknown faces without adding to database
- **Confidence Scoring**: View recognition confidence levels
- **Auto-refresh**: Sidebar data updates automatically after operations

## File Structure

```
face-recognition-attendance/
├── app.py               # Main Streamlit application
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── makefile             # Build and run automation
├── src/
│   ├── database.py      # Database operations and models
│   └── face_utils.py    # Face detection and recognition utilities
└── data/                # Generated at runtime
    ├── faces.db         # SQLite database
    ├── db_images/       # Stored person face images
    └── temp_faces/      # Temporary processing files
```

## Technology Stack

- **Frontend**: Streamlit for web interface
- **Computer Vision**: OpenCV for face detection
- **Face Recognition**: DeepFace for face matching
- **Database**: SQLite for data persistence
- **Image Processing**: PIL, NumPy for image manipulation
- **Build System**: Make for automation

## Configuration

The system can be configured through `config.py`:

```python
# Recognition sensitivity (0.0 - 1.0)
RECOGNITION_THRESHOLD = 0.7

# Minimum face size for detection
FACE_DETECTION_MIN_SIZE = (40, 40)

# Face bounding box appearance
BOUNDING_BOX_OFFSET = 20
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make all` | Setup and run application |
| `make setup` | Install all dependencies |
| `make run` | Run in foreground |
| `make run-background` | Run in background with PID tracking |
| `make stop-app` | Stop background application |
| `make check-app` | Check if application is running |
| `make clean` | Remove all data and logs |
| `make logs` | Create logs directory |
| `make tail-app` | View last application logs |

## Database Schema

### Persons Table
- `id`: Auto-increment primary key
- `name`: Person's name
- `image_path`: Path to stored face image
- `created_at`: Registration timestamp

### Attendance Table
- `id`: Auto-increment primary key
- `person_id`: Foreign key to persons table
- `date`: Attendance date
- `timestamp`: Exact attendance time
- `status`: Attendance status (default: "present")

## Logging

The application supports logging:

- **Setup logs**: `logs/setup.log`
- **Application logs**: `logs/streamlit.log`
- **Timestamped logs**: `logs/streamlit_YYYYMMDD_HHMMSS.log`

## Troubleshooting

### Common Issues

1. **Camera not working**: Ensure camera permissions and check browser settings
2. **Recognition accuracy**: Adjust `RECOGNITION_THRESHOLD` in config.py
3. **Performance issues**: Adjust image size or face detection parameters
4. **Database errors**: Use `make clean` to reset and start fresh

### System Requirements

- **Memory**: 2GB+ RAM recommended
- **CPU**: Multi-core processor for better performance
- **Storage**: 100MB+ for database and images
- **Network**: Not required (runs locally)

## Security Considerations

- Face images are stored locally in the `data/` directory
- Database contains no sensitive personal information beyond names
- No network communication required for core functionality
- Consider data backup and access control for production use

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For bug reports and feature requests, please use the issue tracker.

---

**Note**: This system is designed for educational and small-scale attendance tracking purposes. For enterprise deployment, consider additional security measures, scalability improvements, and compliance requirements.
