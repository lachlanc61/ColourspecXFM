import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
from math import sqrt

import xfmreadout.utils as utils

def data_gaussianblur(data, dims, kernelsize: int):
    """
    applies a gaussian blur to a single map according to kernel size (in pixels, = sd param) 
    """

    """
    BUGHERE: MAP ROLL CANT DIFFERENTIATE X,Y from N, chan
    """

    map = utils.map_roll(data, dims, single=True)

    updated_map = ndimage.gaussian_filter(map, kernelsize, mode='mirror')

    updated_data, dims__ = utils.map_unroll(updated_map)

    return updated_data


def data_resize(data, dims, zoom_factor, order=1):
    """
    resizes a map 

    """
    map = utils.map_roll(data, dims)

    #if multiple channels are present (ie. X, Y, NCHAN)
    #   do not resize along channel axis
    if len(map.shape) == 3:
        zoom = (zoom_factor, zoom_factor, 1)
    elif len(map.shape) <= 2:
        zoom = zoom_factor
    else:
        raise ValueError(f"invalid number of axes for map shape: {map.shape}, expected len() = 1-3")
    
    updated_map = ndimage.zoom(map,  zoom, order=order)     #BUGHERE ndimage is adding a dimension
                                                            #ie. (999,) to (999,1)
        #order 1 = bilinear, 2 = bicubic
    
    updated_data, updated_dims = utils.map_unroll(updated_map)

    return updated_data, updated_dims


def apply_gaussianblur(data, sd_data, dims, kernelsize: int):
    """
    applies a gaussian blur to a map according to kernel size (in pixels, = sd param) 

    updates error maps
    """
    updated_data = data_gaussianblur(data, dims, kernelsize)

    error_factor = sqrt(4^kernelsize)

    updated_sd = sd_data/error_factor   #rough calc

    return updated_data, updated_sd


def apply_resize(data, sd_data, dims, zoom_factor):
    """
    resizes a map / sd pair

    DECPRECATED
    """    
    if zoom_factor < 1:    #downsampling
        order = 1   #bicubic
        error_factor = sqrt(4^order)
    else:
        order = 2   #bilinear
        error_factor = 1    #dont change error if upscaling

    updated_data, updated_dims = data_resize(data, dims, zoom_factor, order=order)
  
    updated_sd = sd_data/error_factor    #rough calc

    return updated_data, updated_sd, updated_dims


