# Posture Analysis System

A real-time posture recognition and analysis system built with Python, OpenCV, and MediaPipe that helps users maintain proper sitting posture while working at a computer.

## Features

- **Real-time posture detection** using computer vision
- **Angle measurements** for neck and torso alignment
- **Color-coded feedback** (green for good posture, red for poor posture)
- **Time tracking** of good vs. poor posture periods
- **Percentage calculation** of correct posture time
- **Visual guides** with angle visualization
- **Web-based interface** accessible from any browser


## Requirements

- Python 3.10 (MediaPipe is not compatible with Python 3.13)
- Webcam
- Required Python packages (see Installation)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/posture-analysis.git
cd posture-analysis
```

2. Create a virtual environment and activate it:
```
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install opencv-python mediapipe flask
```

## Usage

1. Start the application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Position yourself in front of your webcam:
   - Sit up straight with your back against the chair
   - Position your computer so your upper body is visible
   - Ensure good lighting for accurate detection

4. The system will analyze your posture in real-time:
   - **Green lines**: Good posture
   - **Red lines**: Poor posture
   - **Posture status**: Shows whether your current posture is good or poor
   - **Time tracking**: Shows how long you've maintained good and poor posture
   - **Percentage**: Shows the percentage of time spent in good posture

## How It Works

The system uses MediaPipe's pose detection to identify key body landmarks:
- **Neck angle**: Calculated between shoulder and ear
- **Torso angle**: Calculated between hip and shoulder

Good posture is defined as:
- Neck angle < 40 degrees
- Torso angle < 10 degrees

## Technical Details

- **OpenCV**: Used for image processing and drawing visualizations
- **MediaPipe**: Provides the pose estimation model for landmark detection
- **Flask**: Serves the web interface and handles the video stream
- **HTML/CSS**: Frontend interface with responsive design

## File Structure

- `app.py`: Main Flask application
- `camera.py`: Camera handling and posture detection logic
- `templates/index.html`: Web interface template
- `static/`: CSS and JavaScript files

## Troubleshooting

- **No person detected**: Make sure your upper body is clearly visible in the camera view
- **Inaccurate detection**: Ensure you have good lighting and a clear background
- **Camera not working**: Check your camera permissions and connections

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [OpenCV](https://opencv.org/) computer vision library
- [Flask](https://flask.palletsprojects.com/) web framework 