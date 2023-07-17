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
    img_ = ndimage.gaussian_filter(img, kernelsize, mode='mirror')

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
    if len(img.shape) == 3:
        zoom = (zoom_factor, zoom_factor, 1)
    elif len(img.shape) <= 2:
        zoom = zoom_factor
    else:
        raise ValueError(f"invalid number of axes for map shape: {img.shape}, expected len() = 1-3")
    
    updated_map = ndimage.zoom(img,  zoom, order=order)     
        #order 1 = bilinear, 2 = bicubic
    
    updated_data, updated_dims = utils.map_unroll(updated_map)

    return updated_data, updated_dims


def apply_gaussian(img, kernelsize: int, se_=None ):
    """
    applies a gaussian blur

    updates error maps
    """
    if not len(img.shape) == 2:
        raise ValueError("Expected a 2D image of shape Y,X")

    img_ = gaussianblur(img, kernelsize)

    error_factor = sqrt(4^kernelsize)   #rough, not really correct calc

    if se_ is not None:
        se_ = se_/error_factor   

    return img_, se_


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



def calc_quantiles(data, sd, multiplier):
    DATA_QUANTILE=0.999
    SD_QUANTILE_MIN=0.25
    SD_QUANTILE_MAX=0.5

    max_data = np.max(data)
    q99_data = utils.mean_within_quantile(data, qmin=DATA_QUANTILE)

    q2_sd = utils.mean_within_quantile(sd, qmin=SD_QUANTILE_MIN, qmax=SD_QUANTILE_MAX)

    #ratio = q99_data / (q2_sd*multiplier)

    ratio = (q2_sd*multiplier) / q99_data

    return ratio, q99_data, q2_sd