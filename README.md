# Computer Vision Project

## Quick Start

This project features automatic setup and camera detection.

```bash
# 1. Set up your environment (one-time setup)
./setup.sh

# 2. Run the project (auto-detects everything)
sudo $(which python) run.py
```

The project will automatically detect your camera, set up the virtual camera, and handle permissions.

## Features

### Basic Image Processing
- **Statistical Analysis**: Real-time calculation of mean, mode, standard deviation, min/max values for each RGB channel
- **Linear Transformations**: Adjustable brightness and contrast with preset configurations
- **Histogram Processing**: Live histogram display with equalization capabilities
- **Entropy Calculation**: Information content analysis for each color channel

### Advanced Filters
- **Gaussian Blur**: Smoothing filter with configurable kernel size
- **Sharpen Filter**: Edge enhancement using convolution kernel
- **Edge Detection**: Canny edge detection with RGB output
- **Histogram Equalization**: Automatic contrast enhancement

### Special Features
- **Hand Detection**: Real-time hand landmark detection using MediaPipe
- **Vulcan Salute Recognition**: Advanced gesture detection with visual feedback
- **Letter M Detection**: Special gesture that triggers the Herobrine easter egg
- **Live Statistics Overlay**: Real-time display of image statistics and filter status
- **Automatic Setup**: Auto-detects cameras, sets up virtual camera, handles permissions
- **Herobrine Easter Egg**: Special effect triggered by making the letter "M" gesture

## Requirements

- **Python**: 3.10 (managed via pyenv + pyenv-virtualenv)
- **Operating System**: Linux (tested on Arch-based systems), Windows, macOS
- **Additional Software**: OBS Studio (for testing virtual camera functionality)
- **Python Version Manager**: pyenv with pyenv-virtualenv plugin

## Installation

### Automatic Setup (Recommended)

#### 1. Install Python 3.10.12 with pyenv

For more information about how to install pyenv and pyenv-virtualenv, see the documentation for both:
- [pyenv](https://github.com/pyenv/pyenv)
- [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)

```bash
# On Arch
sudo pacman -S pyenv

# Install pyenv-virtualenv as a plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

# Configure shell (add to ~/.zshrc or ~/.bashrc)
eval "$(pyenv init - zsh)"
eval "$(pyenv virtualenv-init -)"
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"

# Reload shell and install Python 3.10
source ~/.zshrc  # or source ~/.bashrc
pyenv install 3.10.12
```

#### 2. Clone and Set Up the Project
```bash
# Clone the repository
git clone git@github.com:viniciusbregoli/computer-vision-project.git
cd computer-vision-project

# Create and activate virtual environment
pyenv virtualenv 3.10.12 computer-vision-project
pyenv activate computer-vision-project
pyenv local computer-vision-project

# Install dependencies
pip install -r requirements.txt

# Run automatic setup
./setup.sh

# Start the project
sudo $(which python) run.py
```

#### 3. If You Need Manual Group Setup
If the setup script couldn't add you to the video group automatically:

```bash
# Add yourself to video group (requires logout/login after)
sudo usermod -a -G video $USER

# Temporary activation (for current session)
newgrp video

# Then run the project
sudo $(which python) run.py
```

### Manual Setup (Fallback)

If automatic setup doesn't work on your system, follow these manual steps:

<details>
<summary>Click to expand manual setup instructions</summary>

#### System-Specific Setup

##### Linux (Arch)
```bash
# Install v4l2loopback for virtual camera support
sudo pacman -S v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="VirtualCam" exclusive_caps=1
```

##### Windows
- Install OBS Studio
- Ensure OBS Virtual Camera is available

##### macOS
- Install OBS Studio or similar virtual camera software

#### Manual Permission Setup
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and back in, or run:
newgrp video
```

#### Manual Camera Detection
```bash
# Test which cameras work
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'Camera {i}: Working')
        cap.release()
"
```

</details>

## Usage

### Starting the Application

The enhanced `run.py` automatically handles:
- **Camera Detection**: Scans and finds working cameras
- **Virtual Camera Setup**: Creates `/dev/video20` (or alternative)
- **Permission Handling**: Sets up X11 access when needed
- **Error Recovery**: Provides helpful troubleshooting messages

```bash
# Simple start (recommended)
sudo $(which python) run.py

# If you get permission errors:
newgrp video && sudo $(which python) run.py

# As a last resort:
sudo sudo $(which python) run.py
```

### Troubleshooting

If you encounter issues:

```bash
# 1. Run the setup script again
./setup.sh

# 2. Check system status
python -c "
import cv2
print('OpenCV version:', cv2.__version__)
# Test camera access
cap = cv2.VideoCapture(1)
print('Camera 1 available:', cap.isOpened())
cap.release()
"

# 3. Check video devices
ls -la /dev/video*

# 4. Check user groups
groups | grep video

# 5. Manual virtual camera setup
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="VirtualCam" exclusive_caps=1
```

## Controls

| Key | Function |
|-----|----------|
| `n` | Toggle Hand Detection |
| `g` | Toggle Debug Mode (detailed gesture detection info) |
| `h` | Toggle Histogram Display |
| `s` | Toggle Statistics Overlay |
| `b` | Toggle Blur Filter |
| `p` | Toggle Sharpen Filter |
| `d` | Toggle Edge Detection |
| `e` | Toggle Histogram Equalization |
| `t` | Toggle Linear Transformation |
| `1-4` | Linear transformation presets (when transform is active) |
| `q` | Quit Application |

### Linear Transformation Presets
- **1**: Identity (a=1.0, b=0) - Original image
- **2**: Brighten (a=1.5, b=50) - Increased brightness and contrast
- **3**: Darken (a=0.7, b=-20) - Reduced brightness
- **4**: High Contrast (a=2.0, b=0) - Maximum contrast

## Hand Detection Features

The hand detection system uses MediaPipe for real-time tracking.

### Gesture Detection
The system can detect two special gestures:

#### Vulcan Salute
- Analyzes finger positioning and separation
- Validates proper finger grouping (index+middle, ring+pinky)
- Provides visual feedback with green bounding box
- Displays "LIVE LONG AND PROSPER :D" message when detected

#### Letter M Gesture
- Detects when fingers form the letter "M" shape
- Triggers the Herobrine easter egg
- Darkens the screen and displays Herobrine image
- Shows "herobrine is here" message

![Hand Detection Demo](demo/image.png)
![Vulcan Salute Detection](demo/image2.png)

## Configuration

### Automatic Configuration
The application automatically configures:
- **Camera Selection**: Uses first working camera found
- **Resolution**: Attempts 1280x720, falls back to camera default
- **Virtual Camera**: Creates `/dev/video20` or next available device
- **Frame Rate**: Targets 30fps

### Manual Configuration (if needed)
Edit `run.py` to override automatic settings:

```python
# In main() function, modify these values:
width = 1280      # Output resolution width
height = 720      # Output resolution height
fps = 30          # Target frame rate

# Force specific camera (overrides auto-detection):
# vc.capture_cv_video(0, bgr_to_rgb=True)  # Force camera 0
```

### Hand Detection Parameters
Modify `HandDetector` initialization in `run.py`:
```python
hand_detector = HandDetector(
    max_num_hands=2,                    # Maximum hands to track
    min_detection_confidence=0.7,       # Detection threshold
    min_tracking_confidence=0.5,        # Tracking threshold
    debug=False                         # Debug output
)
```

## File Structure

```
computer-vision-project/
├── run.py                  # Main application (enhanced with auto-setup)
├── setup.sh               # Automatic system setup script
├── capturing.py            # Camera and virtual camera handling
├── basics.py              # Basic image processing functions
├── hand_detection.py       # MediaPipe hand detection and gesture recognition
├── overlays.py            # Image overlay utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── demo/                  # Demo images
    ├── image.png
    └── image2.png
```

## Getting Help

If you run into issues:

1. **Run the setup script**: `./setup.sh`
2. **Check the troubleshooting section** in this README
3. **Look at the automatic error messages** - they provide specific guidance
4. **Try running with different permissions**: `newgrp video` or `sudo`
5. **Verify your camera works** with other applications first

The enhanced version is designed to handle most setup issues automatically and provide clear guidance when manual intervention is needed.