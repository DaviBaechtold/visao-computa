# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: Davi Baechtold Campos, Vinicius Bregoli, Lincoln Rodrigo
"""
from numba import njit  # conda install numba
import numpy as np
import math


@njit
def histogram_figure_numba(np_img):
    """
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
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

    # Max and min
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
        if p_i_r > 0: # log2(0) is undefined
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


####

### All other basic functions

####
