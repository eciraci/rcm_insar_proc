#!/usr/bin/env python
u"""
Written by Enrico Ciraci' - 11/2020
Fill NoData values in a 2D array using GDAL's FillNodata function.

More info here:
https://gdal.org/api/gdal_alg.html#_CPPv414GDALFillNodata15GDALRaster
    BandH15GDALRasterBandHdiiPPc16GDALProgressFuncPv
"""
import numpy as np
from osgeo import gdal
from osgeo import gdal_array


def fill_nodata(data_arr: np.ndarray, mask_arr: np.ndarray,
                max_search_dist: int = 1000,
                smth_iter: int = 10) -> np.ndarray:
    """
    Fill NoData values in a 2D array using GDAL's FillNodata function
    :param data_arr: input 2D array [numpy ndarray]
    :param mask_arr: mask array [numpy ndarray]
    :param max_search_dist: maximum search distance [int]
    :param smth_iter: smoothing iterations [int]
    :return: filled array [numpy ndarray]
    """
    # - extract input array dimensions
    src_ds = gdal_array.OpenArray(data_arr)
    srcband = src_ds.GetRasterBand(1)
    dstband = srcband
    mask_ds = gdal_array.OpenArray(mask_arr)
    maskband = mask_ds.GetRasterBand(1)
    options = []
    gdal.FillNodata(dstband, maskband, max_search_dist, smth_iter,
                    options, callback=None)
    out_arr = dstband.ReadAsArray()
    return out_arr


