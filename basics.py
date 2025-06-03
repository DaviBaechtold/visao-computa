# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: Davi Baechtold Campos, Vinicius Bregoli, Lincoln Rodrigo
"""
from numba import njit  # conda install numba
import numpy as np
import math
import cv2


@njit
def histogram_figure_numba(np_img):
    """
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    """
    r_bars = [0] * 256
    g_bars = [0] * 256
    b_bars = [0] * 256

    for j in range(np_img.shape[0]):
        for k in range(np_img.shape[1]):
            pixel = np_img[j, k]
            redIntensity = int(pixel[0])
            greenIntensity = int(pixel[1])
            blueIntensity = int(pixel[2])
            r_bars[redIntensity] += 1
            g_bars[greenIntensity] += 1
            b_bars[blueIntensity] += 1

    return r_bars, g_bars, b_bars


def calculate_channels_stats(r_bars, g_bars, b_bars):
    # Mean = sum of intensities / n of pixels
    r_mean = 0
    g_mean = 0
    b_mean = 0
    for i in range(256):
        r_mean += i * r_bars[i]
        g_mean += i * g_bars[i]
        b_mean += i * b_bars[i]
    r_sum = sum(r_bars)
    g_sum = sum(g_bars)
    b_sum = sum(b_bars)
    r_mean /= r_sum
    g_mean /= g_sum
    b_mean /= b_sum

    # Mode = most common intensity
    r_mode = r_bars.index(max(r_bars))
    g_mode = g_bars.index(max(g_bars))
    b_mode = b_bars.index(max(b_bars))

    # Standard Deviation = sqrt(sum of (intensity - mean)^2 / n of pixels)
    std_r = 0
    std_g = 0
    std_b = 0
    for i in range(256):
        std_r += (i - r_mean) ** 2 * r_bars[i]
        std_g += (i - g_mean) ** 2 * g_bars[i]
        std_b += (i - b_mean) ** 2 * b_bars[i]
    std_r = (std_r / r_sum) ** 0.5
    std_g = (std_g / g_sum) ** 0.5
    std_b = (std_b / b_sum) ** 0.5

    # Max and min of each channel
    for i in range(255, -1, -1):
        if r_bars[i] > 0:
            r_max = i
            break
    for i in range(255, -1, -1):
        if g_bars[i] > 0:
            g_max = i
            break
    for i in range(255, -1, -1):
        if b_bars[i] > 0:
            b_max = i
            break
    for i in range(256):
        if r_bars[i] > 0:
            r_min = i
            break
    for i in range(256):
        if g_bars[i] > 0:
            g_min = i
            break
    for i in range(256):
        if b_bars[i] > 0:
            b_min = i
            break

    # Entropy = -sum of (p_i * log2(p_i))
    r_entropy = 0
    g_entropy = 0
    b_entropy = 0
    for i in range(256):
        p_i_r = r_bars[i] / r_sum
        p_i_g = g_bars[i] / g_sum
        p_i_b = b_bars[i] / b_sum
        if p_i_r > 0:  # log2(0) is undefined
            r_entropy += p_i_r * math.log2(p_i_r)
        if p_i_g > 0:
            g_entropy += p_i_g * math.log2(p_i_g)
        if p_i_b > 0:
            b_entropy += p_i_b * math.log2(p_i_b)
    r_entropy = -r_entropy
    g_entropy = -g_entropy
    b_entropy = -b_entropy

    return (
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
    )


def linear_transformation(np_img, a, b):
    lookup_table = np.arange(256)  # LUT for optimization
    lookup_table = lookup_table * a + b  # Linear transformation
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)  # Clip to 0-255
    return lookup_table[np_img]  # Apply LUT to image


def equalize_histogram(r_bars, g_bars, b_bars):
    """
    Creates lookup tables for histogram equalization
    """
    # Calculate the cumulative distribution function
    cdf_r = np.cumsum(r_bars)
    cdf_g = np.cumsum(g_bars)
    cdf_b = np.cumsum(b_bars)

    # Create equalized lookup tables
    # Avoid division by zero by checking if cdf[-1] > 0
    lookup_table_r = np.zeros(256, dtype=np.uint8)
    lookup_table_g = np.zeros(256, dtype=np.uint8)
    lookup_table_b = np.zeros(256, dtype=np.uint8)

    if cdf_r[-1] > 0:
        lookup_table_r = (cdf_r * 255 / cdf_r[-1]).astype(np.uint8)
    if cdf_g[-1] > 0:
        lookup_table_g = (cdf_g * 255 / cdf_g[-1]).astype(np.uint8)
    if cdf_b[-1] > 0:
        lookup_table_b = (cdf_b * 255 / cdf_b[-1]).astype(np.uint8)

    return lookup_table_r, lookup_table_g, lookup_table_b


def apply_histogram_equalization(
    np_img, lookup_table_r, lookup_table_g, lookup_table_b
):
    """
    Apply histogram equalization lookup tables to an image
    """
    # Create a copy of the image to avoid modifying the original
    equalized_img = np_img.copy()

    # Apply lookup tables to each channel separately
    equalized_img[:, :, 0] = lookup_table_r[np_img[:, :, 0]]  # Red channel
    equalized_img[:, :, 1] = lookup_table_g[np_img[:, :, 1]]  # Green channel
    equalized_img[:, :, 2] = lookup_table_b[np_img[:, :, 2]]  # Blue channel

    return equalized_img


def blur_filter(np_img):
    # Parameters
    # (25, 25) is the kernel size
    # 0 is the standard deviation in the X direction
    return cv2.GaussianBlur(np_img, (25, 25), 0)


def sharpen_filter(np_img):
    # Using kernel
    # [-1, -1, -1]
    # [-1,  9, -1]
    # [-1, -1, -1]
    return cv2.filter2D(np_img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def edge_detection(np_img):
    edges = cv2.Canny(np_img, 50, 150)
    # Convert to uint8 RGB format
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_rgb.astype(np.uint8)
