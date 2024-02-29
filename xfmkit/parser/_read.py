
import time
import numpy as np


import parsercore

import xfmkit.bufferops as bufferops
import xfmkit.utils as utils
import xfmkit.structures as structures

from ._parse import *
from ._utils import *

import logging
logger = logging.getLogger(__name__)

class MapDone(Exception): pass


def read(config, args, dirs):
    """
    Parse full file, creating map and extracted data objects
    """
    #start a timer
    starttime = time.time() 
    
    try:
        #initialise map object
        xfmap = structures.Xfmap(config, dirs.fi, dirs.fsub, args.write_modified, args.chunk_size, args.multiload)

        #initialise the spectrum-by-pixel object
        pixelseries = structures.PixelSeries(config, xfmap, xfmap.npx, xfmap.detarray, args.index_only)

        pixelseries, xfmap.indexlist = indexmap(xfmap, pixelseries, args.multiload)

        if not args.index_only:
            pixelseries = parse(xfmap, pixelseries, args.multiload)
            pixelseries = pixelseries.get_derived()    #calculate additional derived properties after parse

        #assign modified deadtimes
        if not args.modify_deadtimes > 100: #-1 = False
            pixelseries = pixelseries.get_dtmod(config, xfmap, args.modify_deadtimes)

        if args.write_modified:
            writemap(config, xfmap, pixelseries, args.x_coords, args.y_coords, \
                args.modify_deadtimes, args.multiload)

    finally:
        xfmap.closefiles()

        #complete the timer
        runtime = time.time() - starttime

        print(
        "---------------------------\n"
        "PARSING COMPLETE\n"
        "---------------------------\n"
        f"dimensions expected (x,y): {xfmap.xres},{xfmap.yres}\n"
        f"pixels expected (X*Y): {xfmap.npx}\n"
        f"pixels found: {pixelseries.npx}\n"
        f"total time: {round(runtime,2)} s\n"
        f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
        "---------------------------"
        )

        #export the pixel header stats and data

        pixelseries.exportpxstats(config, dirs.exports)

        if args.export_data:
            pixelseries.exportpxdata(config, dirs.exports)    

    return xfmap, pixelseries