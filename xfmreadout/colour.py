import numpy as np
import os
import sys
#import pybaselines.smooth  #for background fitting

#from colorsys import hsv_to_rgb

import xfmreadout.utils as utils
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#-----------------------------------
#MODIFIABLE CONSTANTS
#-----------------------------------

#-----------------------------------
#INITIALISE
#-----------------------------------

# create a pointer to the module object instance itself
#       functions like "self" for module
#   -> more explicit than just using module local namespace
#   https://stackoverflow.com/questions/1977362/how-to-create-module-wide-variables-in-python
this = sys.modules[__name__]

#-----------------------------------
#FUNCTIONS
#-----------------------------------

def initialise(config, e):
    """
    initialise the colour gaussians 
    export to module-wide variables via "this"

    receives energy channel list
    returns None

    
    NB: not certain this is optimal - just need to get e somehow
        could also create in parallel via config.NCHAN & ESTEP
    """
    kachan=utils.lookfor(e,float(config['ELASTIC']+config['EOFFSET']))  #K-alpha channel
    npad=config['NCHAN']-kachan

    hsv=cm.get_cmap('hsv', kachan)
    cmap=hsv(range(kachan))

    cred=cmap[:,0]
    cgreen=cmap[:,1]
    cblue=cmap[:,2]

    cred=np.pad(cred, (0, npad), 'constant', constant_values=(0, 1))
    cgreen=np.pad(cgreen, (0, npad), 'constant', constant_values=(0, 0))
    cblue=np.pad(cblue, (0, npad), 'constant', constant_values=(0, 0))    

    #assign to module-wide variables
    this.red=cred
    this.green=cgreen
    this.blue=cblue

    return None

def spectorgb(config, e, y):
    """
    maps spectrum onto R G B channels 
    use HSV colourmap to generate
    """
    #bg = pybaselines.smooth.snip(y,SNIPWINDOW)[0]
    #y=y-bg

    #if doing log y
    if config['RGBLOG']:
        #convert y to float for log
        yf=y.astype(float)
        #log y, excluding 0 values
        y=np.log(yf, out=np.zeros_like(yf), where=(yf!=0))

    #multiply y vectorwise onto channels (t/px: 0.004051 s)
    rsum=np.sum(y*(this.red))/len(e)
    gsum=np.sum(y*(this.green))/len(e)
    bsum=np.sum(y*(this.blue))/len(e)

    ysum=np.sum(y)
    
#    max=np.max([rsum,bsum,gsum])

    return(rsum,gsum,bsum,ysum)

def complete(rvals, gvals, bvals, mapx, mapy, dirs):
    """
    creates final colour-mapped image

    recives R G B arrays per pixel, and total counts per pixel

    displays plot
    """
    print(f'rgb maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')
    chmax=np.max([np.max(rvals),np.max(gvals),np.max(bvals)])
    rvals=rvals/chmax
    gvals=gvals/chmax
    bvals=bvals/chmax

    print(f'scaled maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')

    np.savetxt(os.path.join(dirs.transforms, "colourmap_red.txt"), rvals)
    np.savetxt(os.path.join(dirs.transforms, "colourmap_green.txt"), gvals)
    np.savetxt(os.path.join(dirs.transforms, "colourmap_blue.txt"), bvals)

    rimg=np.reshape(rvals, (-1, mapx))
    gimg=np.reshape(gvals, (-1, mapx))
    bimg=np.reshape(bvals, (-1, mapx))

    rgbarray = np.zeros((mapy,mapx,3), 'uint8')
    rgbarray[..., 0] = rimg*256
    rgbarray[..., 1] = gimg*256
    rgbarray[..., 2] = bimg*256
    
    show(rgbarray, dirs.exports)
    return(rgbarray)


def show(rgbarray, out_path):
    plt.imshow(rgbarray)
    plt.savefig(os.path.join(out_path, 'colours.png'), dpi=150)
