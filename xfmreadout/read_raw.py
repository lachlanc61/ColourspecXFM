import time
import sys

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.colour as colour
import xfmreadout.clustering as clustering
import xfmreadout.visualisations as vis
import xfmreadout.structures as structures
import xfmreadout.dtops as dtops
import xfmreadout.parser as parser
import xfmreadout.diagops as diagops

"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- extracts pixel data
- classifies data via eg. UMAP, HDBSCAN
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

def entry():
    """
    entry point without explicit args
    """
    args_in = sys.argv[1:]  #NB: exclude 0 == script name
    main(args_in)

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

    #perform parse
    xfmap, pixelseries = parser.read(config, args, dirs)

    #ANALYSIS

    #perform post-analysis:
    #   create and show colourmap, deadtime/sum reports
    if args.analyse:
        #uncomment to fit baselines
        #pixelseries.corrected=fitting.calc_corrected(pixelseries.flattened, pixelseries.energy, pixelseries.npx, pixelseries.nchan)

        #dtops.export(dirs.exports, pixelseries.dtpred, pixelseries.flatsum)

        if args.log_file is not None:
            realtime, livetime, triggers, events, icr, ocr, dt_evt, dt_rt = diagops.dtfromdiag(dirs.logf)
            print(dt_evt)

        dtops.dtplots(config, dirs.plots, pixelseries.dt, pixelseries.sum, pixelseries.dtpred[:,0], pixelseries.dtflat, \
            pixelseries.flatsum, xfmap.xres, xfmap.yres, pixelseries.ndet, args.index_only)

        pixelseries.rgbarray, pixelseries.rvals, pixelseries.gvals, pixelseries.bvals \
            = colour.calccolours(config, pixelseries, xfmap, pixelseries.flattened, dirs)       #flattened / corrected
    else:
        rgbarray = None
    #perform clustering
    if args.classify_spectra:
        pixelseries.categories, pixelseries.classavg, embedding, clusttimes = clustering.run( pixelseries.flattened, dirs.transforms, force_embed=args.force, overwrite=config['OVERWRITE_EXPORTS'] )
        
        vis.plot_clusters(pixelseries.categories, pixelseries.classavg, embedding, pixelseries.dimensions)
#        clustering.complete(pixelseries.categories, pixelseries.classavg, embedding, clusttimes, xfmap.energy, xfmap.xres, xfmap.yres, config['nclust'], dirs.plots)
        #colour.plot_colourmap_explainer(pixelseries.energy, pixelseries.classavg[1:1], pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, dirs)
    else:
        categories = None
        classavg = None

    print("Processing complete")

    return pixelseries, xfmap, #dt_log

if __name__ == "__main__":
    args_in = sys.argv[1:]

    main(args_in)        

    sys.exit()