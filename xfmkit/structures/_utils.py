import os
import numpy as np
from scipy import ndimage

import xfmkit.bufferops as bufferops
import xfmkit.dtops as dtops
import xfmkit.imgops as imgops
import xfmkit.utils as utils
import xfmkit.config as config

import logging
logger = logging.getLogger(__name__)

def data_unroll(maps):
    """
    reshape map (x, y, counts) to data (i, counts)

    returns dataset and dimensions
    """

    if len(maps.shape) == 3:
        data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
        dims=maps.shape[:2]
    elif len(maps.shape) == 2:
        data=maps.reshape(maps.shape[0]*maps.shape[1])
        dims=maps.shape[:2]        
    else:
        raise ValueError(f"unexpected dimensions for {map}")

    return data, dims    