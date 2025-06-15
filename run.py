# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard
import numpy as np
import cv2  # Added for text overlay
import os
import sys
import subprocess
import platform
from capturing import VirtualCamera
from overlays import (
    initialize_hist_figure,
    plot_overlay_to_image,
    plot_strings_to_image,
    update_histogram,
)
from basics import (
    histogram_figure_numba,
    calculate_channels_stats,
    linear_transformation,
    equalize_histogram,
    apply_histogram_equalization,
    blur_filter,
    sharpen_filter,
    edge_detection,
)
from hand_detection import HandDetector

# Load Herobrine image once (after other imports, before main loop)
herobrine_img = cv2.imread('herobrine.jpg', cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to keep alpha if present



def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at (x, y) with alpha_mask."""
    h, w = img_overlay.shape[:2]
    alpha = alpha_mask / 255.0
    for c in range(0, 3):
        img[y:y+h, x:x+w, c] = (1. - alpha) * img[y:y+h, x:x+w, c] + alpha * img_overlay[:, :, c]
    return img

class SystemSetup:
    """Handles automatic system setup and camera detection across platforms"""
    
    @staticmethod
    def get_platform():
        """Detect the current platform"""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "macos"
        elif system == "linux":
            return "linux"
        else:
            return "unknown"
    
    @staticmethod
    def detect_working_camera():
        """Auto-detect the first working camera"""
        print("🔍 Auto-detecting cameras...")
        
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"✅ Found working camera at index {i} ({width}x{height})")
                        cap.release()
                        return i
                    cap.release()
            except Exception:
                continue
        
        print("❌ No working cameras found!")
        return None
    
    @staticmethod
    def setup_virtual_camera_linux(video_nr=20):
        """Set up v4l2loopback virtual camera on Linux"""
        print("🔧 Setting up v4l2loopback virtual camera...")
        
        # Check if already loaded with correct parameters
        virtual_device = f"/dev/video{video_nr}"
        if os.path.exists(virtual_device):
            print(f"✅ Virtual camera already exists: {virtual_device}")
            return video_nr
        
        # Check if v4l2loopback module exists
        try:
            result = subprocess.run(['modinfo', 'v4l2loopback'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ v4l2loopback module not available. Install with:")
                print("   # Arch Linux:")
                print("   sudo pacman -S v4l2loopback-dkms")
                print("   # Ubuntu/Debian:")
                print("   sudo apt install v4l2loopback-dkms")
                return None
        except Exception:
            print("❌ Cannot check v4l2loopback module")
            return None
        
        # Try to load the module
        try:
            # First, try to unload existing module
            subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], 
                         capture_output=True)
            
            # Load with our parameters
            cmd = ['sudo', 'modprobe', 'v4l2loopback', 
                   'devices=1', f'video_nr={video_nr}', 
                   'card_label=VirtualCam', 'exclusive_caps=1']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(virtual_device):
                print(f"✅ Virtual camera created: {virtual_device}")
                return video_nr
            else:
                print(f"⚠️  Failed to create virtual camera at video{video_nr}")
                # Try alternative video numbers
                for alt_nr in [3, 4, 5, 21, 22]:
                    try:
                        subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], 
                                     capture_output=True)
                        cmd = ['sudo', 'modprobe', 'v4l2loopback', 
                               'devices=1', f'video_nr={alt_nr}', 
                               'card_label=VirtualCam', 'exclusive_caps=1']
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        alt_device = f"/dev/video{alt_nr}"
                        if result.returncode == 0 and os.path.exists(alt_device):
                            print(f"✅ Virtual camera created: {alt_device}")
                            return alt_nr
                    except Exception:
                        continue
                
                print("❌ Could not create virtual camera on any device")
                return None
                
        except Exception as e:
            print(f"❌ Error setting up virtual camera: {e}")
            print("   Try running: sudo modprobe v4l2loopback devices=1 video_nr=20")
            return None
    
    @staticmethod
    def setup_virtual_camera_windows():
        """Set up virtual camera on Windows (requires OBS)"""
        print("🔧 Checking Windows virtual camera setup...")
        
        # Check if OBS is installed by looking for common installation paths
        obs_paths = [
            r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
            r"C:\Program Files (x86)\obs-studio\bin\32bit\obs32.exe",
            os.path.expanduser(r"~\AppData\Local\obs-studio\bin\64bit\obs64.exe"),
        ]
        
        obs_found = False
        for path in obs_paths:
            if os.path.exists(path):
                print(f"✅ Found OBS Studio at: {path}")
                obs_found = True
                break
        
        if not obs_found:
            print("❌ OBS Studio not found. Please install OBS Studio:")
            print("   1. Download from: https://obsproject.com/")
            print("   2. Install OBS Studio")
            print("   3. Start OBS, click 'Start Virtual Camera', then 'Stop Virtual Camera'")
            print("   4. Close OBS and run this script again")
            return None
        
        # Test if pyvirtualcam can find the OBS virtual camera
        try:
            import pyvirtualcam
            # Try to create a camera to test if OBS virtual camera is available
            with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
                print("✅ OBS Virtual Camera is available and working")
                return "obs"
        except Exception as e:
            print("⚠️  OBS Virtual Camera not properly set up:")
            print(f"   Error: {e}")
            print("   To fix:")
            print("   1. Start OBS Studio")
            print("   2. Click 'Start Virtual Camera' (bottom right panel)")
            print("   3. Click 'Stop Virtual Camera'")
            print("   4. Close OBS and try again")
            return None
    
    @staticmethod
    def setup_virtual_camera_macos():
        """Set up virtual camera on macOS (requires OBS)"""
        print("🔧 Checking macOS virtual camera setup...")
        
        # Check if OBS is installed
        obs_path = "/Applications/OBS.app"
        if not os.path.exists(obs_path):
            print("❌ OBS Studio not found. Please install OBS Studio:")
            print("   1. Download from: https://obsproject.com/")
            print("   2. Install OBS Studio")
            print("   3. Start OBS, click 'Start Virtual Camera', then 'Stop Virtual Camera'")
            print("   4. Close OBS and run this script again")
            return None
        
        print(f"✅ Found OBS Studio at: {obs_path}")
        
        # Test if pyvirtualcam can find the OBS virtual camera
        try:
            import pyvirtualcam
            # Try to create a camera to test if OBS virtual camera is available
            with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
                print("✅ OBS Virtual Camera is available and working")
                return "obs"
        except Exception as e:
            print("⚠️  OBS Virtual Camera not properly set up:")
            print(f"   Error: {e}")
            print("   To fix:")
            print("   1. Start OBS Studio")
            print("   2. Click 'Start Virtual Camera' (bottom right panel)")
            print("   3. Click 'Stop Virtual Camera'")
            print("   4. Close OBS and try again")
            
            # Check for macOS 14+ compatibility issues
            macos_version = platform.mac_ver()[0]
            if macos_version and float('.'.join(macos_version.split('.')[:2])) >= 14.0:
                print("   ⚠️  Note: macOS 14+ has known compatibility issues with pyvirtualcam")
                print("   Consider using an older macOS version or contributing to fix:")
                print("   https://github.com/letmaik/pyvirtualcam/issues/111")
            
            return None
    
    @staticmethod
    def check_permissions_linux():
        """Check and fix permissions on Linux"""
        print("🔐 Checking Linux permissions...")
        
        # Get current user info
        if os.geteuid() == 0:  # Running as root
            print("⚠️  Running as root - some GUI features may not work")
            # Try to set up X11 access
            try:
                original_user = os.environ.get('SUDO_USER')
                if original_user:
                    subprocess.run(['xhost', f'+si:localuser:{original_user}'], 
                                 capture_output=True)
                    print("✅ X11 access configured for GUI components")
            except Exception:
                print("⚠️  Could not configure X11 access - histograms may not display")
            return True
        else:
            # Check if user is in video group
            try:
                import pwd
                import grp
                username = pwd.getpwuid(os.getuid()).pw_name
                user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
                
                if 'video' in user_groups:
                    print("✅ User has video group permissions")
                    return True
                else:
                    print("⚠️  User not in video group")
                    print(f"   Run: sudo usermod -a -G video {username}")
                    print("   Then log out and back in, or run with 'newgrp video'")
                    return False
            except ImportError:
                print("⚠️  Cannot check group membership (missing modules)")
                return True
    
    @staticmethod
    def check_permissions_windows():
        """Check permissions on Windows"""
        print("🔐 Checking Windows permissions...")
        print("✅ Windows permissions should be handled automatically")
        return True
    
    @staticmethod
    def check_permissions_macos():
        """Check permissions on macOS"""
        print("🔐 Checking macOS permissions...")
        print("✅ macOS permissions should be handled automatically")
        print("   Note: You may need to grant camera access in System Preferences")
        return True
    
    @staticmethod
    def auto_setup():
        """Perform complete automatic setup based on platform"""
        current_platform = SystemSetup.get_platform()
        
        print("🚀 Computer Vision Project - Cross-Platform Auto Setup")
        print("=" * 55)
        print(f"🖥️  Detected platform: {current_platform.title()}")
        
        # Check permissions based on platform
        if current_platform == "linux":
            permissions_ok = SystemSetup.check_permissions_linux()
        elif current_platform == "windows":
            permissions_ok = SystemSetup.check_permissions_windows()
        elif current_platform == "macos":
            permissions_ok = SystemSetup.check_permissions_macos()
        else:
            print("⚠️  Unknown platform - proceeding with basic setup")
            permissions_ok = True
        
        # Detect working camera (works on all platforms)
        camera_index = SystemSetup.detect_working_camera()
        if camera_index is None:
            print("\n❌ Setup failed: No working camera found")
            print("Make sure:")
            print("- Camera is connected and recognized")
            print("- Camera is not in use by another application")
            print("- User has proper permissions")
            return None, None, current_platform
        
        # Setup virtual camera based on platform
        virtual_camera_info = None
        if current_platform == "linux":
            virtual_camera_info = SystemSetup.setup_virtual_camera_linux()
        elif current_platform == "windows":
            virtual_camera_info = SystemSetup.setup_virtual_camera_windows()
        elif current_platform == "macos":
            virtual_camera_info = SystemSetup.setup_virtual_camera_macos()
        else:
            print("⚠️  Virtual camera setup not supported on this platform")
        
        print(f"\n✅ Setup complete!")
        print(f"   Platform: {current_platform.title()}")
        print(f"   Camera: index {camera_index}")
        
        if virtual_camera_info:
            if current_platform == "linux":
                print(f"   Virtual camera: /dev/video{virtual_camera_info}")
            else:
                print(f"   Virtual camera: OBS Virtual Camera (backend: {virtual_camera_info})")
        else:
            print("   Virtual camera: Not available (application will work without it)")
        
        return camera_index, virtual_camera_info, current_platform


class KeyPressDetector:  # This class is used to detect if a key is just pressed
    def __init__(self):
        self.previous_states = {}

    def is_key_just_pressed(self, key):
        current_state = keyboard.is_pressed(key)
        previous_state = self.previous_states.get(key, False)

        # Update the stored state for next frame
        self.previous_states[key] = current_state

        # Return True only on the transition from False to True
        return current_state and not previous_state


def custom_processing(img_source_generator):
    key_detector = KeyPressDetector()
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    # Initialize hand detector
    hand_detector = HandDetector(
        max_num_hands=2, min_detection_confidence=0.7, debug=False
    )

    SHOW_STATS = True
    SHOW_HIST = False
    SHOW_TRANSFORM = False
    SHOW_EQUALIZATION = False
    SHOW_BLUR = False
    SHOW_SHARPEN = False
    SHOW_EDGE_DETECTION = False
    SHOW_HAND_DETECTION = False

    a = 1
    b = 0

    # Debounce/hysteresis variables for background application
    m_detected_counter = 0
    m_not_detected_counter = 0
    background_applied = False
    FRAMES_TO_CONFIRM = 5  # Number of consecutive frames to confirm gesture

    try:
        for sequence in img_source_generator:
            # Handle keyboard input
            if key_detector.is_key_just_pressed("h"):
                SHOW_HIST = not SHOW_HIST
            if key_detector.is_key_just_pressed("s"):
                SHOW_STATS = not SHOW_STATS
            if key_detector.is_key_just_pressed("t"):
                SHOW_TRANSFORM = not SHOW_TRANSFORM
            if key_detector.is_key_just_pressed("e"):
                SHOW_EQUALIZATION = not SHOW_EQUALIZATION
            if key_detector.is_key_just_pressed("b"):
                SHOW_BLUR = not SHOW_BLUR
            if key_detector.is_key_just_pressed("p"):
                SHOW_SHARPEN = not SHOW_SHARPEN
            if key_detector.is_key_just_pressed("d"):
                SHOW_EDGE_DETECTION = not SHOW_EDGE_DETECTION
            if key_detector.is_key_just_pressed("n"):  # 'n' for haNd detection
                SHOW_HAND_DETECTION = not SHOW_HAND_DETECTION
            if key_detector.is_key_just_pressed("g"):  # 'g' for debuG mode
                hand_detector.debug = not hand_detector.debug
                print(f"Debug mode: {'ON' if hand_detector.debug else 'OFF'}")
            if key_detector.is_key_just_pressed("q"): # 'q' for Quit
                print("\n👋 Quitting program...")
                sys.exit(0)
            # Filters (apply in order)

            # Blur filter
            if SHOW_BLUR:
                sequence = blur_filter(sequence)

            # Sharpen filter
            if SHOW_SHARPEN:
                sequence = sharpen_filter(sequence)

            # Edge detection
            if SHOW_EDGE_DETECTION:
                sequence = edge_detection(sequence)

            # Linear transformation controls
            if SHOW_TRANSFORM:
                if key_detector.is_key_just_pressed("1"):
                    a, b = 1.0, 0
                if key_detector.is_key_just_pressed("2"):
                    a, b = 1.5, 50
                if key_detector.is_key_just_pressed("3"):
                    a, b = 0.7, -20
                if key_detector.is_key_just_pressed("4"):
                    a, b = 2.0, 0
                sequence = linear_transformation(sequence, a, b)

            # Handle equalization with before/after histogram display
            if SHOW_EQUALIZATION:
                # Get histogram of current image for equalization calculation
                r_bars_before, g_bars_before, b_bars_before = histogram_figure_numba(
                    sequence
                )

                # Apply equalization
                r_lookup_table, g_lookup_table, b_lookup_table = equalize_histogram(
                    r_bars_before, g_bars_before, b_bars_before
                )
                sequence = apply_histogram_equalization(
                    sequence, r_lookup_table, g_lookup_table, b_lookup_table
                )

            # Hand detection and landmark drawing
            vulcan_detected = False
            letter_m_detected = False
            if SHOW_HAND_DETECTION:
                sequence, vulcan_detected, letter_m_detected = hand_detector.detect_and_draw_hands(
                    sequence
                )

            # Get final histogram for display and stats
            if SHOW_HIST or SHOW_STATS:
                r_bars, g_bars, b_bars = histogram_figure_numba(sequence)

            # Display histogram of final result
            if SHOW_HIST:
                max_r = max(r_bars) if max(r_bars) > 0 else 1
                max_g = max(g_bars) if max(g_bars) > 0 else 1
                max_b = max(b_bars) if max(b_bars) > 0 else 1

                r_bars_norm = [i * 3 / max_r for i in r_bars]
                g_bars_norm = [i * 3 / max_g for i in g_bars]
                b_bars_norm = [i * 3 / max_b for i in b_bars]

                update_histogram(
                    fig,
                    ax,
                    background,
                    r_plot,
                    g_plot,
                    b_plot,
                    r_bars_norm,
                    g_bars_norm,
                    b_bars_norm,
                )
                sequence = plot_overlay_to_image(sequence, fig)

            # Display stats
            if SHOW_STATS:
                (
                    r_mean,
                    g_mean,
                    b_mean,
                    r_mode,
                    g_mode,
                    b_mode,
                    std_r,
                    std_g,
                    std_b,
                    r_max,
                    g_max,
                    b_max,
                    r_min,
                    g_min,
                    b_min,
                    r_entropy,
                    g_entropy,
                    b_entropy,
                ) = calculate_channels_stats(r_bars, g_bars, b_bars)

                display_text_arr = [
                    f"Mean(RGB): ({r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f})",
                    f"Mode(RGB): ({r_mode}, {g_mode}, {b_mode})",
                    f"Std(RGB): ({std_r:.2f}, {std_g:.2f}, {std_b:.2f})",
                    f"Max(RGB): ({r_max}, {g_max}, {b_max})",
                    f"Min(RGB): ({r_min}, {g_min}, {b_min})",
                    f"Entropy(RGB): ({r_entropy:.2f}, {g_entropy:.2f}, {b_entropy:.2f})",
                ]
                # Add status indicators for active filters
                if SHOW_EQUALIZATION:
                    display_text_arr.append("Histogram Equalized: ON")
                if SHOW_TRANSFORM:
                    display_text_arr.append(f"Linear Transformation: a={a}, b={b}")
                if SHOW_BLUR:
                    display_text_arr.append("Blur Filter: ON")
                if SHOW_SHARPEN:
                    display_text_arr.append("Sharpen Filter: ON")
                if SHOW_EDGE_DETECTION:
                    display_text_arr.append("Edge Detection: ON")
                if SHOW_HAND_DETECTION:
                    display_text_arr.append("Hand Detection: ON")
                    if vulcan_detected:
                        display_text_arr.append("VULCAN SALUTE DETECTED! 🖖")
                    elif letter_m_detected:
                        display_text_arr.append("LETTER M DETECTED! 📝")
                    else:
                        display_text_arr.append("No special gestures detected")

                # Add control instructions
                display_text_arr.append("--- Controls ---")
                display_text_arr.append("'n': Toggle Hand Detection")
                display_text_arr.append("'g': Toggle Debug Mode")
                display_text_arr.append("'h': Toggle Histogram")
                display_text_arr.append("'s': Toggle Stats")
                display_text_arr.append("'b': Toggle Blur")
                display_text_arr.append("'p': Toggle Sharpen")
                display_text_arr.append("'d': Toggle Edge Detection")
                display_text_arr.append("'e': Toggle Equalization")
                display_text_arr.append("'t': Toggle Transform")

                sequence = plot_strings_to_image(sequence, display_text_arr)

            if letter_m_detected:
                # Darken the whole frame
                sequence = (sequence * 0.4).astype(np.uint8)
                if herobrine_img is not None:
                    # Resize Herobrine image if needed
                    scale = 0.2  # 20% of original size
                    h_h, w_h = int(herobrine_img.shape[0] * scale), int(herobrine_img.shape[1] * scale)
                    herobrine_small = cv2.resize(herobrine_img, (w_h, h_h), interpolation=cv2.INTER_AREA)

                    # Where to place (bottom right)
                    h_frame, w_frame = sequence.shape[:2]
                    x_offset = w_frame - w_h - 10  # 10px from right
                    y_offset = h_frame - h_h - 10  # 10px from bottom

                    # If image has alpha channel
                    if herobrine_small.shape[2] == 4:
                        alpha_mask = herobrine_small[:, :, 3]
                        sequence = overlay_image_alpha(sequence, herobrine_small[:, :, :3], x_offset, y_offset, alpha_mask)
                    else:
                        # No alpha, just overlay
                        sequence[y_offset:y_offset+h_h, x_offset:x_offset+w_h] = herobrine_small

                    # Add text at the top center
                    text = "herobrine is here"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.0
                    thickness = 4
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = (w_frame - text_size[0]) // 2
                    text_y = 60  # 60px from the top
                    cv2.putText(
                        sequence,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )

            yield sequence

    finally:
        # Clean up hand detector resources
        hand_detector.close()


def main():
    # Perform automatic setup
    camera_index, virtual_camera_info, current_platform = SystemSetup.auto_setup()
    
    if camera_index is None:
        print("\n❌ Cannot start application - setup failed")
        sys.exit(1)
    
    # Default settings
    width = 1280
    height = 720
    fps = 30

    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)

    print(f"\n🎬 Starting Computer Vision Project")
    print("Controls:")
    print("'n' - Toggle Hand Detection")
    print("'g' - Toggle Debug Mode (prints gesture detection details)")
    print("'h' - Toggle Histogram")
    print("'s' - Toggle Stats")
    print("'b' - Toggle Blur Filter")
    print("'p' - Toggle Sharpen Filter")
    print("'d' - Toggle Edge Detection")
    print("'e' - Toggle Histogram Equalization")
    print("'t' - Toggle Linear Transformation")
    print("'1-4' - Different transformation presets (when transform is ON)")
    print("'q' - Quit")
    print("\n🎥 Camera feed starting...")

    # Prepare virtual camera interaction arguments
    interaction_kwargs = {}
    
    if virtual_camera_info:
        if current_platform == "linux":
            interaction_kwargs['device'] = f"/dev/video{virtual_camera_info}"
        # For Windows/macOS, pyvirtualcam automatically finds OBS virtual camera
    
    try:
        vc.virtual_cam_interaction(
            custom_processing(
                vc.capture_cv_video(camera_index, bgr_to_rgb=True)
            ),
            **interaction_kwargs
        )
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        print("\nTroubleshooting:")
        print("- Make sure no other application is using the camera")
        
        if current_platform == "windows":
            print("- Ensure OBS Studio is installed and virtual camera is set up")
            print("- Try starting OBS, enable virtual camera, then stop it")
        elif current_platform == "macos":
            print("- Ensure OBS Studio is installed and virtual camera is set up") 
            print("- Check System Preferences for camera permissions")
            print("- Note: macOS 14+ has known compatibility issues")
        else:  # Linux
            print("- Try running with sudo if permission issues persist")
            print("- Check that v4l2loopback is installed")
        
        sys.exit(1)


if __name__ == "__main__":
    main()