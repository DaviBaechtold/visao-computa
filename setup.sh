#!/bin/bash

# Computer Vision Project Setup Script
# This script sets up everything needed to run the project

echo "🚀 Computer Vision Project Setup"
echo "================================"

# Check if running on supported system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Detected Linux system"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  Detected macOS - some features may need manual setup"
else
    echo "⚠️  Detected Windows/Other - manual setup may be required"
fi

# Function to check if user is in video group
check_video_group() {
    if groups | grep -q video; then
        echo "✅ User is already in video group"
        return 0
    else
        echo "❌ User is not in video group"
        return 1
    fi
}

# Function to add user to video group
add_to_video_group() {
    echo "🔧 Adding user to video group..."
    sudo usermod -a -G video $USER
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully added user to video group"
        echo "⚠️  You need to log out and back in, or run 'newgrp video'"
        return 0
    else
        echo "❌ Failed to add user to video group"
        return 1
    fi
}

# Function to install v4l2loopback on Arch-based systems
install_v4l2loopback_arch() {
    echo "🔧 Installing v4l2loopback for Arch Linux..."
    
    # Check if already installed
    if pacman -Q v4l2loopback-dkms >/dev/null 2>&1; then
        echo "✅ v4l2loopback-dkms already installed"
        return 0
    fi
    
    # Install it
    sudo pacman -S --noconfirm v4l2loopback-dkms
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed v4l2loopback-dkms"
        return 0
    else
        echo "❌ Failed to install v4l2loopback-dkms"
        return 1
    fi
}

# Function to check Python virtual environment
check_python_env() {
    echo "🐍 Checking Python environment..."
    
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$PYENV_VERSION" ]]; then
        echo "✅ Virtual environment detected: ${VIRTUAL_ENV:-$PYENV_VERSION}"
        
        # Check Python version
        python_version=$(python --version 2>&1)
        echo "   Python version: $python_version"
        
        # Check if required packages are installed
        if python -c "import cv2, mediapipe, numpy, matplotlib" 2>/dev/null; then
            echo "✅ Required Python packages are installed"
            return 0
        else
            echo "⚠️  Some required packages may be missing"
            echo "   Run: pip install -r requirements.txt"
            return 1
        fi
    else
        echo "⚠️  No virtual environment detected"
        echo "   Recommended: Use pyenv virtualenv or venv"
        return 1
    fi
}

# Function to test camera access
test_camera() {
    echo "📹 Testing camera access..."
    
    python3 -c "
import cv2
found_camera = False
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'✅ Camera {i}: Working')
            found_camera = True
        cap.release()
        break

if not found_camera:
    print('❌ No working cameras found')
    exit(1)
" 2>/dev/null

    return $?
}

# Main setup process
main() {
    echo
    echo "Starting automatic setup..."
    echo
    
    # 1. Check Python environment
    check_python_env
    python_ok=$?
    
    # 2. Check/fix video group membership
    if ! check_video_group; then
        read -p "Add user to video group? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            add_to_video_group
            video_group_added=true
        else
            echo "⚠️  Skipping video group setup"
            video_group_added=false
        fi
    else
        video_group_added=false
    fi
    
    # 3. Install v4l2loopback if on Arch
    if command -v pacman >/dev/null 2>&1; then
        read -p "Install/check v4l2loopback? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_v4l2loopback_arch
        fi
    fi
    
    # 4. Test camera access
    test_camera
    camera_ok=$?
    
    echo
    echo "📋 Setup Summary:"
    echo "=================="
    
    if [ $python_ok -eq 0 ]; then
        echo "✅ Python environment: OK"
    else
        echo "⚠️  Python environment: Needs attention"
    fi
    
    if check_video_group; then
        echo "✅ Video group: OK"
    else
        echo "⚠️  Video group: Needs logout/login or 'newgrp video'"
    fi
    
    if [ $camera_ok -eq 0 ]; then
        echo "✅ Camera access: OK"
    else
        echo "❌ Camera access: Issues detected"
    fi
    
    echo
    echo "🎬 Ready to run!"
    echo "================"
    
    if [ $video_group_added = true ]; then
        echo "⚠️  IMPORTANT: You were added to the video group."
        echo "   Please run one of these commands:"
        echo "   - newgrp video  (temporary, for this session)"
        echo "   - Log out and back in (permanent)"
        echo
        echo "Then run: python run.py"
    else
        echo "Run your project with:"
        echo "   python run.py"
        echo
        echo "If you get permission errors, try:"
        echo "   newgrp video && python run.py"
        echo "   OR"
        echo "   sudo python run.py"
    fi
}

# Run main function
main