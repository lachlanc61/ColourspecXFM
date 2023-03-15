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
import xfmreadout.fitting as fitting
import xfmreadout.diagops as diagops

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
PACKAGE_CONFIG='xfmreadout/config.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

def main(args_in):
    """
    parse map according to args_in
    
    return pixelseries, xfmap and analysis results
    """
    #create input config from args and config files
    config =utils.initcfg(PACKAGE_CONFIG)

    #get command line arguments
    args = argops.readargs(args_in, config)

    #initialise read file and directory structure 
    config, dirs = utils.initfiles(args, config)

    #start a timer
    starttime = time.time() 
    
    try:
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
        #uncomment to fit baselines
        #pixelseries.corrected=fitting.calc_corrected(pixelseries.flattened, pixelseries.energy, pixelseries.npx, pixelseries.nchan)

        #dtops.export(dirs.exports, pixelseries.dtpred, pixelseries.flatsum)

        if args.log_file is not None:
            realtime, livetime, triggers, events, icr, ocr, dt_evt, dt_rt = diagops.dtfromdiag(dirs.logf)
            print(dt_evt)

        dtops.dtplots(config, dirs.plots, pixelseries.dt, pixelseries.sum, pixelseries.dtpred[0], pixelseries.dtflat, \
            pixelseries.flatsum, xfmap.xres, xfmap.yres, pixelseries.ndet, args.index_only)

        pixelseries.rgbarray, pixelseries.rvals, pixelseries.gvals, pixelseries.bvals \
            = colour.calccolours(config, pixelseries, xfmap, pixelseries.flattened, dirs)       #flattened / corrected
    else:
        rgbarray = None
    #perform clustering
    if args.classify_spectra:
        pixelseries.categories, pixelseries.classavg = clustering.complete(config, pixelseries.corrected, xfmap.energy, xfmap.npx, xfmap.xres, xfmap.yres, dirs)
        #colour.plot_colourmap_explainer(pixelseries.energy, pixelseries.classavg[1:1], pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, dirs)
    else:
        categories = None
        classavg = None

    print("Processing complete")

    return pixelseries, xfmap, #dt_log

if __name__ == "__main__":
    main(sys.argv[1:])      #NB: exclude 0 == script name

    sys.exit()