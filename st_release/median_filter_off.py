#!/usr/bin/env python
"""
Written by Enrico Ciraci' - 11/2020
Originally written by Jeremie Mouginot - 2018

Generate Offsets Outliers Mask by employing a median filter.
"""
import numpy as np
from scipy.signal import medfilt


def median_filter_off(off: np.ndarray, size: int = 9,
                      thre: int = 3) -> np.ndarray:
    """
    Median filter for offset array
    :param off: offsets array [numpy ndarray - complex]
    :param size: median filter size [int]
    :param thre: median filter threshold [int]
    :return: outliers mask [numpy ndarray]
    """

    vram = medfilt(off.real, size)
    vazm = medfilt(off.imag, size)

    mask = (np.abs(off.real - vram) > thre) | (
                np.abs(off.imag - vazm) > thre) | (off.imag == 0) | (
                       off.real == 0)

    return mask
