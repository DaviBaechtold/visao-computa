# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard

from capturing import VirtualCamera
from overlays import (
    initialize_hist_figure,
    plot_overlay_to_image,
    plot_strings_to_image,
    update_histogram,
)
from basics import histogram_figure_numba, calculate_channels_stats


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    SHOW_HIST = True
    SHOW_STATS = True

    for sequence in img_source_generator:
        # sequence is a numpy array of shape (height, width, 3)
        # Call your custom processing methods here! (e. g. filters)

        # Example of keyboard is pressed
        if keyboard.is_pressed("h"):
            SHOW_HIST = not SHOW_HIST
        if keyboard.is_pressed("s"):
            SHOW_STATS = not SHOW_STATS

        # Load the histogram values
        if SHOW_HIST:
            r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        if SHOW_STATS:
            # Calculate the stats
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

        # Normalize them between 0 and 3
        if SHOW_HIST:
            r_bars = [i * 3 / max(r_bars) for i in r_bars]
            g_bars = [i * 3 / max(g_bars) for i in g_bars]
            b_bars = [i * 3 / max(b_bars) for i in b_bars]

        # Update the histogram with new data
        if SHOW_HIST:
            update_histogram(
                fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars
            )
            # uses the figure to create the overlay
            sequence = plot_overlay_to_image(sequence, fig)

        # Display text example
        if SHOW_STATS:
            display_text_arr = [
                f"Mean(RGB): ({r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f})",
                f"Mode(RGB): ({r_mode}, {g_mode}, {b_mode})",
                f"Std(RGB): ({std_r:.2f}, {std_g:.2f}, {std_b:.2f})",
                f"Max(RGB): ({r_max}, {g_max}, {b_max})",
                f"Min(RGB): ({r_min}, {g_min}, {b_min})",
                f"Entropy(RGB): ({r_entropy:.2f}, {g_entropy:.2f}, {b_entropy:.2f})",
            ]
            sequence = plot_strings_to_image(sequence, display_text_arr)

        # Make sure to yield your processed image
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
        )
    )


if __name__ == "__main__":
    main()
