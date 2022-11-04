import time
import sys
import numpy as np
import xfmreadout.utils as utils
import xfmreadout.colour as colour
import xfmreadout.clustering as clustering
import xfmreadout.obj as obj
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

#initialise read file and all directories relative to current script
config, fi, fname, fsub, odir = utils.initf(config)

starttime = time.time()             #init timer

#-----------------------------------
#MAIN START
#-----------------------------------

#initialise map object
#   parses header into map.headerdict
#   places pointer (map.idx) at start of first pixel record
xfmap = obj.Xfmap(config, fi, fsub)

detarray=xfmap.getdetectors(config)

#initialise the spectrum-by-pixel object
#       pre-creates all arrays for storing data, pixel header values etc
#       WARNING: big memory spike here if map is large
pixelseries = obj.PixelSeries(config, xfmap)

#start a timer
starttime = time.time() 

try:
    #if we are parsing the .GeoPIXE file
    #   begin parsing
    if config['FORCEPARSE']:
        pixelseries = xfmap.parse(config, pixelseries)
    #else read from a pre-parsed csv
    else:   
        pixelseries = pixelseries.readseries(config, odir)
finally:
    xfmap.closefiles()

runtime = time.time() - starttime

print(
"---------------------------\n"
"MAP COMPLETE\n"
"---------------------------\n"
f"pixels expected (X*Y): {xfmap.numpx}\n"
f"pixels found: {pixelseries.npx}\n"
f"total time: {round(runtime,2)} s\n"
f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
"---------------------------"
)

pixelseries.exportheader(config, odir)

if not config['PARSEMAP']:
    print("WRITE COMPLETE")
    print("---------------------------")
    exit()

if config['SAVEPXSPEC']:
    pixelseries.exportseries(config, odir)

#show memory usage
utils.varsizes(locals().items())

#perform post-analysis:
UDET=config['use_detector'] #define working detector for multi-detector files

#create and show colour map
if config['DOCOLOURS'] == True:

    colour.initialise(config, xfmap.energy)
    
    for i in np.arange(pixelseries.npx):
        counts=pixelseries.data[UDET,i,:]
        pixelseries.rvals[i], pixelseries.bvals[i], pixelseries.gvals[i], pixelseries.totalcounts[i] = colour.spectorgb(config, xfmap.energy, counts)

    rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, xfmap.xres, pixelseries.nrows, odir)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, pixelseries.data[UDET], xfmap.energy, xfmap.numpx, xfmap.xres, xfmap.yres, odir)

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
    read+write:             0.000643 s
"""