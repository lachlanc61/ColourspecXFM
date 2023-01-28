import time
import sys
import numpy as np
import xfmreadout.utils as utils
import xfmreadout.colour as colour
import xfmreadout.clustering as clustering
import xfmreadout.obj as obj
import xfmreadout.dtops as dtops
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
USER_CONFIG='config.yaml'
PACKAGE_CONFIG='xfmreadout/protocol.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

#get command line arguments
args = utils.readargs()

#create input config from args and config files
config, rawconfig=utils.initcfg(args, PACKAGE_CONFIG, USER_CONFIG)

#initialise read file and directory structure 
config, dirs = utils.initfiles(config)

starttime = time.time()             #init timer

#-----------------------------------
#MAIN START
#-----------------------------------

#initialise map object
#   parses header into map.headerdict
#   places pointer (map.idx) at start of first pixel record
xfmap = obj.Xfmap(config, dirs.fi, dirs.fsub)

#initialise the spectrum-by-pixel object
#       pre-creates all arrays for storing data, pixel header values etc
#       WARNING: big memory spike here if map is large
pixelseries = obj.PixelSeries(config, xfmap, xfmap.numpx, xfmap.detarray)

#start a timer
starttime = time.time() 

parser.indexmap(xfmap, pixelseries)



exit()


try:
    #if we are parsing the .GeoPIXE file
    #   begin parsing
    if config['FORCEPARSE']:
        pixelseries = xfmap.parse(config, pixelseries)
    #else read from a pre-parsed csv
    else:   
        pixelseries = pixelseries.readpxdata(config, dirs.exports)
finally:
    xfmap.closefiles(config)

runtime = time.time() - starttime

#show memory usage
utils.varsizes(locals().items())

print(
"---------------------------\n"
"MAP COMPLETE\n"
"---------------------------\n"
f"dimensions expected (x,y): {xfmap.xres},{xfmap.yres}\n"
f"pixels expected (X*Y): {xfmap.numpx}\n"
f"pixels found: {pixelseries.npx}\n"
f"total time: {round(runtime,2)} s\n"
f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
"---------------------------"
)

pixelseries.exportpxstats(config, dirs.exports)

if not config['PARSEMAP']:
    print("WRITE COMPLETE")
    print("---------------------------")
    exit()

if config['SAVEPXSPEC']:
    pixelseries.exportpxdata(config, dirs.exports)

#perform post-analysis:

UDET=config['use_detector'] #define working detector for multi-detector files

#generate deadtime/sum reports
if config['DODTCALCS'] == True:
    dtpred, dtavg, mergedsum = dtops.postcalc(config, pixelseries, xfmap)

    dtops.export(dirs.exports, dtpred, mergedsum)

    dtops.dtplots(config, dirs.plots, pixelseries.dt, pixelseries.sum, dtpred, dtavg, mergedsum, xfmap.xres, xfmap.yres, pixelseries.ndet)


#create and show colour map
if config['DOCOLOURS'] == True:

    colour.initialise(config, xfmap.energy)
    
    for i in np.arange(pixelseries.npx):
        counts=pixelseries.data[UDET,i,:]
        pixelseries.rvals[i], pixelseries.bvals[i], pixelseries.gvals[i], pixelseries.totalcounts[i] = colour.spectorgb(config, xfmap.energy, counts)

    rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, xfmap.xres, pixelseries.nrows, dirs)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, pixelseries.data[UDET], xfmap.energy, xfmap.numpx, xfmap.xres, xfmap.yres, dirs)

print("Processing complete")

sys.exit()

"""
runtime log:
                            t/px
reading only:               0.000140 s
+clustering                 0.001296 s     
colourmap:                  0.007800 s

improving colourmap:    
    for j:                  0.007625 s
    vectorise channels:     0.004051 s
    pre-init gaussians:     0.002641 s   
    fully vectorised:       0.001886 s

w/ background fitting:
    snip:                   0.002734 s
    complex snip:           0.002919 s

OO:
    map+pxseries:           0.001852 s
    chunk parsing:          0.002505 s

refactored:
    parse+write:             0.000643 s

dtops:
    read+parse              0.001184 s
    headers+write           0.00005 s
"""