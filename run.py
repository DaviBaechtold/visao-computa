# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard
import numpy as np
import cv2  # Added for text overlay
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
            if SHOW_HAND_DETECTION:
                sequence, vulcan_detected = hand_detector.detect_and_draw_hands(
                    sequence
                )

                # Add prominent Vulcan salute indicator
                if vulcan_detected:
                    h, w, _ = sequence.shape
                    # Add large text overlay
                    cv2.putText(
                        sequence,
                        "LIVE LONG AND PROSPER :D",
                        (w // 2 - 300, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        4,
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
                    else:
                        display_text_arr.append("Vulcan Salute: Not detected")

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

            yield sequence

    finally:
        # Clean up hand detector resources
        hand_detector.close()


def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30

    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)

    print("Starting Computer Vision Project with Hand Detection")
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

    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            # or your window screen
            # vc.capture_screen()
        ),
        device="/dev/video20",
    )


if __name__ == "__main__":
    main()
