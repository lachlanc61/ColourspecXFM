import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
from math import sqrt

import xfmreadout.utils as utils

def map_gaussianblur(map, kernelsize: int):
    """
    applies a gaussian blur to a map according to kernel size (in pixels, = sd param) 
    """

    modified_map = ndimage.gaussian_filter(map, kernelsize, mode='mirror')

    error_factor = sqrt(4^kernelsize)

    return modified_map, error_factor


def apply_gaussianblur(data, sd_data, dims, kernelsize: int):
    """
    applies a gaussian blur to a map according to kernel size (in pixels, = sd param) 

    updates error maps
    """
    map = utils.map_roll(data, dims)

    modified_map, error_factor = map_gaussianblur(map, kernelsize)

    updated_data = utils.map_unroll(modified_map)

    updated_sd = sd_data/error_factor   #rough calc

    return updated_data, updated_sd


def map_resize(map, zoom_factor, order=1):
    """
    resizes a map 
    """
    #order 1 = bilinear, 2 = bicubic
    result = ndimage.zoom(map,  zoom_factor, order=order)
    
    if zoom_factor <= 1:
        error_factor = sqrt(4^order)
    else: 
        error_factor = 1    #dont change error if upscaling

    return result, error_factor


def apply_resize(data, sd_data, dims, zoom_factor):
    """
    resizes a map 
    """    
    map = utils.map_roll(data, dims)

    if zoom_factor <= 1:
        order = 1   #bicubic
    else:
        order = 2   #bilinear

    modified_map, error_factor = map_resize(map, zoom_factor, order=order)

    updated_data = utils.map_unroll(modified_map)

    updated_dims = modified_map.shape

    updated_sd = sd_data/error_factor    #rough calc

    return updated_data, updated_sd, updated_dims


