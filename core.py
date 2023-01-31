import time
import sys
import numpy as np
import xfmreadout.utils as utils
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

#-----------------------------------
#MAIN START
#-----------------------------------

try:
    #start a timer
    starttime = time.time() 

    #initialise map object
    xfmap = structures.Xfmap(config, dirs.fi, dirs.fsub)

    #initialise the spectrum-by-pixel object
    pixelseries = structures.PixelSeries(config, xfmap, xfmap.npx, xfmap.detarray)

    pixelseries, indexlist = parser.indexmap(xfmap, pixelseries)

    if config['PARSEMAP']:
        pixelseries = parser.parse(xfmap, pixelseries, indexlist)

    if config['WRITESUBMAP']:
        parser.writemap(config, xfmap, pixelseries)

finally:
    xfmap.closefiles(config)

    #complete the timer
    runtime = time.time() - starttime

print(
"---------------------------\n"
"MAP COMPLETE\n"
"---------------------------\n"
f"dimensions expected (x,y): {xfmap.xres},{xfmap.yres}\n"
f"pixels expected (X*Y): {xfmap.npx}\n"
f"pixels found: {pixelseries.npx}\n"
f"total time: {round(runtime,2)} s\n"
f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
"---------------------------"
)

#calculate derived pixel properties - eg. sums, flattened

pixelseries.exportpxstats(config, dirs.exports)

if config['SAVEPXSPEC']:
    pixelseries.exportpxdata(config, dirs.exports)

#perform post-analysis:

UDET=config['use_detector'] #define working detector for multi-detector files

#generate deadtime/sum reports
if config['DODTCALCS'] == True:

    dtpred, dtavg = dtops.postcalc(config, pixelseries, xfmap)

    dtops.export(dirs.exports, dtpred, pixelseries.flatsum)

    dtops.dtplots(config, dirs.plots, pixelseries.dt, pixelseries.sum, dtpred, dtavg, pixelseries.flatsum, xfmap.xres, xfmap.yres, pixelseries.ndet)

#create and show colour map
if config['DOCOLOURS'] == True:
    colour.initialise(config, xfmap.energy)
    
    for i in np.arange(pixelseries.npx):
        counts=pixelseries.flattened[i,:]
        pixelseries.rvals[i], pixelseries.bvals[i], pixelseries.gvals[i], pixelseries.totalcounts[i] = colour.spectorgb(config, xfmap.energy, counts)

    rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, xfmap.xres, pixelseries.nrows, dirs)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, pixelseries.flattened, xfmap.energy, xfmap.npx, xfmap.xres, xfmap.yres, dirs)

print("Processing complete")

sys.exit()

"""
runtime log:
test on: ts2.GeoPIXE, hw
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
    write only:             0.000643 s
main:                                       #147424d 26/11/2022
    parse                   0.001215

dtops:                                      #65c5eac 27/01/23     
    parse                   0.001184 s
    headers+write           0.00005 s

parallelize - single proc:                  #f3ea355 30/01/23
    index+parse             0.001219 s
    index+parse+write       0.001293 s

parallelize - multiproc:
    index+parse             
"""