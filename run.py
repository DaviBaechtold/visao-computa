# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard
import numpy as np
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

    SHOW_STATS = True
    SHOW_HIST = False
    SHOW_TRANSFORM = False
    SHOW_EQUALIZATION = False
    SHOW_BLUR = False
    SHOW_SHARPEN = False
    SHOW_EDGE_DETECTION = False

    a = 1
    b = 0

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

        # Filters

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
            # Add equalization status to display
            if SHOW_EQUALIZATION:
                display_text_arr.append("Histogram Equalized: ON")
            if SHOW_TRANSFORM:
                display_text_arr.append(f"Linear Transformation: a={a}, b={b}")
            if SHOW_BLUR:
                display_text_arr.append("Blur Filter: ON")
            if SHOW_SHARPEN:
                display_text_arr.append("Sharpen Filter: ON")

            sequence = plot_strings_to_image(sequence, display_text_arr)

        yield sequence


def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30

    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)

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
