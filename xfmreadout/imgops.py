import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
from math import sqrt

import xfmreadout.utils as utils

def gaussianblur(img, kernelsize: int):
    """
    applies a gaussian blur to a single image according to kernel size (in pixels, = sd param) 
    """

    img_ = ndimage.gaussian_filter(map, kernelsize, mode='mirror')

    return img_


def img_resize(img, zoom_factor, order=1):

    img_ = ndimage.zoom(map,  zoom_factor, order=order) 

    return img_

def apply_resize(img, sd_img, zoom_factor, order=1):
    """
    resizes a map 

    """
    
    #if multiple channels are present (ie. X, Y, NCHAN)
    #   do not resize along channel axis
    if len(map.shape) == 3:
        zoom = (zoom_factor, zoom_factor, 1)
    elif len(map.shape) <= 2:
        zoom = zoom_factor
    else:
        raise ValueError(f"invalid number of axes for map shape: {map.shape}, expected len() = 1-3")
    
    updated_map = ndimage.zoom(map,  zoom, order=order)     
        #order 1 = bilinear, 2 = bicubic
    
    updated_data, updated_dims = utils.map_unroll(updated_map)

    return updated_data, updated_dims


def apply_gaussian(img, kernelsize: int, sd_img=None, ):
    """
    applies a gaussian blur

    updates error maps
    """
    img_ = gaussianblur(img, kernelsize)

    error_factor = sqrt(4^kernelsize)   #rough, not really correct calc

    sd_ = sd_img/error_factor   

    return img_, sd_


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


