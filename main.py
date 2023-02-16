import time
import sys
import numpy as np
import argparse

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.colour as colour
import xfmreadout.clustering as clustering
import xfmreadout.structures as structures
import xfmreadout.dtops as dtops
import xfmreadout.bufferops as bufferops
import xfmreadout.parser as parser

"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- extracts pixel data
- classifies data via PCA and UMAP
- displays classified maps
- produces average spectrum per class

./data has example datasets
"""
#-----------------------------------
#vars
#-----------------------------------
PACKAGE_CONFIG='xfmreadout/protocol.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

def main():
    #get the arguments from command line
    argparser = argparse.ArgumentParser(
        description="XFM data loader and analysis package"
    )

    #get command line arguments
    args = argops.readargs(argparser)

    """
    to do:

    adjust initcfg to ignore config in main dir
    -> move any remaining flags from config to protocol
    rename protocol to config

    also work out how to give default args via VSCode...
    """

    #create input config from args and config files
    config, rawconfig=utils.initcfg(args, PACKAGE_CONFIG)

    #initialise read file and directory structure 
    config, dirs = utils.initfiles(args, config)

    #-----------------------------------
    #MAIN START
    #-----------------------------------

    try:
        #start a timer
        starttime = time.time() 

        #initialise map object
        xfmap = structures.Xfmap(config, dirs.fi, dirs.fsub, args.write_modified, args.chunk_size, args.multiprocess)

        #initialise the spectrum-by-pixel object
        pixelseries = structures.PixelSeries(config, xfmap, xfmap.npx, xfmap.detarray, args.index_only)

        pixelseries, indexlist = parser.indexmap(xfmap, pixelseries, args.multiprocess)

        if not args.index_only:
            pixelseries = parser.parse(xfmap, pixelseries, indexlist, args.multiprocess)
            pixelseries = pixelseries.get_derived(config, xfmap)    #calculate additional derived properties after parse

        if args.write_modified:
            parser.writemap(config, xfmap, pixelseries, args.x_coords, args.y_coords, \
                args.fill_deadtimes, args.multiprocess)

    finally:
        xfmap.closefiles()

        #complete the timer
        runtime = time.time() - starttime

    print(
    "---------------------------\n"
    "COMPLETE\n"
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

    #perform post-analysis:
    #   create and show colourmap, deadtime/sum reports
    if args.analyse:
        #dtops.export(dirs.exports, pixelseries.dtpred, pixelseries.flatsum)

        dtops.dtplots(config, dirs.plots, pixelseries.dt, pixelseries.sum, pixelseries.dtpred[0], pixelseries.dtflat, \
            pixelseries.flatsum, xfmap.xres, xfmap.yres, pixelseries.ndet, args.index_only)

        colour.initialise(config, xfmap.energy)
        
        for i in np.arange(pixelseries.npx):
            counts=pixelseries.flattened[i,:]
            pixelseries.rvals[i], pixelseries.bvals[i], pixelseries.gvals[i], pixelseries.totalcounts[i] = colour.spectorgb(config, xfmap.energy, counts)

        rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, xfmap.xres, pixelseries.nrows, dirs)

    #perform clustering
    if args.classify_spectra:
        categories, classavg = clustering.complete(config, pixelseries.flattened, xfmap.energy, xfmap.npx, xfmap.xres, xfmap.yres, dirs)

    print("Processing complete")

if __name__ == "__main__":
    main()

sys.exit()