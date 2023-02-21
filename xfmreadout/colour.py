import numpy as np
import os
import sys
#import pybaselines.smooth  #for background fitting

#from colorsys import hsv_to_rgb

import xfmreadout.utils as utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

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

custom_colour_dict = \
        {'red':   [[0.0,  1.0, 1.0],
                   [0.143,  1.0, 1.0],
                   [0.286,  1.0, 1.0],
                   [0.429,  0.0, 0.0],
                   [0.571,  0.0, 0.0],
                   [0.714,  0.0, 0.0],
                   [0.857,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],

         'green': [[0.0,  0.0, 0.0],
                   [0.143,  0.0, 0.0],
                   [0.286,  1.0, 1.0],
                   [0.429,  1.0, 1.0],
                   [0.571,  1.0, 1.0],
                   [0.714,  0.0, 0.0],
                   [0.857,  0.0, 0.0],
                   [1.0,  1.0, 1.0]],
                   
         'blue':  [[0.0,  1.0, 1.0],
                   [0.143,  0.0, 0.0],
                   [0.286,  0.0, 0.0],
                   [0.429,  0.0, 0.0],
                   [0.571,  1.0, 1.0],
                   [0.714,  1.0, 1.0],
                   [0.857,  1.0, 1.0],
                   [1.0,  1.0, 1.0]]}



#-----------------------------------
#FUNCTIONS
#-----------------------------------

def initialise(config, energy):
    """
    initialise the colourmap

    receives energy channel list
    returns red, green, blue arrays
    """
    
    chan_lo=100
    chan_hi=np.where(energy==config['ELASTIC'])[0][0]

    cmap = matplotlib.colors.LinearSegmentedColormap('hsv_white', segmentdata=custom_colour_dict, N=chan_hi)

    rgba = cmap(np.linspace(0, 1, chan_hi-chan_lo))

    red=rgba[:,0]
    green=rgba[:,1]
    blue=rgba[:,2]

    pad_hi=len(energy)-chan_hi
    pad_lo=chan_lo

    red=np.pad(red, (pad_lo, pad_hi), mode='constant', constant_values=(1, 1))
    green=np.pad(green, (pad_lo, pad_hi), mode='constant', constant_values=(0, 1))
    blue=np.pad(blue, (pad_lo, pad_hi), mode='constant', constant_values=(1, 1))    

    return red, green, blue

def spectorgb(energy, spectrum, red, green, blue):
    """
    maps spectrum onto R G B channels 
    use RGBA colourmap to generate
    """

    #multiply y vectorwise onto channels (t/px: 0.004051 s)
    rsum=np.sum(spectrum*(red))/len(energy)
    gsum=np.sum(spectrum*(green))/len(energy)
    bsum=np.sum(spectrum*(blue))/len(energy)

    ysum=np.sum(spectrum)
    
#    max=np.max([rsum,bsum,gsum])

    return(rsum,gsum,bsum,ysum)

def compile(rvals, gvals, bvals, mapx, mapy):
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

    rimg=np.reshape(rvals, (-1, mapx))
    gimg=np.reshape(gvals, (-1, mapx))
    bimg=np.reshape(bvals, (-1, mapx))

    rgbimg = np.zeros((mapy,mapx,3), 'uint8')
    rgbimg[..., 0] = rimg*256
    rgbimg[..., 1] = gimg*256
    rgbimg[..., 2] = bimg*256
    
    return(rgbimg, rvals, gvals, bvals)


def plot_colourmap_explainer(energy, spectrum, red, green, blue, dirs):
    """
    displays a single spectrum coloured by final map and overlaid with R G B channel multipliers
    """
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('energy (kV)') 
    ax1.set_ylabel('Intensity') 
    ax1.axis(xmin=0,xmax=30)
    ax1.set_yscale('log')
    #ax1.axis(ymin=vymin,ymax=1)
    ax1.axis(ymin=0.00005,ymax=1)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('RGB')
    ax2.axis(ymin=0,ymax=1.05)

    ax2.plot(energy, red, '#ff0000', linestyle='dashed', linewidth=1, label="R")
    ax2.plot(energy, green, '#00ff00', linestyle='dashed', linewidth=1, label="G")
    ax2.plot(energy, blue, '#0000ff', linestyle='dashed', linewidth=1, label="B")

    ax2.legend(loc=(0.85,0.7))

    ax1.plot(energy, spectrum, color="gray")
    for i in range(len(spectrum) - 1):
        ax1.fill_between([energy[i], energy[i+1]], [spectrum[i], spectrum[i+1]], y2=-1, color=(red[i],green[i],blue[i]))


    plt.savefig(os.path.join(dirs.plots, 'rgba_spectrum.png'), dpi=150)
    plt.show()

def export_show(rgbimg, rvals, gvals, bvals, dirs):
    """
    saves colourmap and displays image
    """

    np.savetxt(os.path.join(dirs.transforms, "colourmap_red.txt"), rvals, delimiter=',')
    np.savetxt(os.path.join(dirs.transforms, "colourmap_green.txt"), gvals, delimiter=',')
    np.savetxt(os.path.join(dirs.transforms, "colourmap_blue.txt"), bvals, delimiter=',')

    plt.imshow(rgbimg)
    plt.savefig(os.path.join(dirs.plots, 'colours.png'), dpi=150)


def calccolours(config, pixelseries, xfmap, dataset, dirs):
    red, green, blue = initialise(config, pixelseries.energy)
    
    rvals=np.zeros(pixelseries.npx)
    gvals=np.zeros(pixelseries.npx)
    bvals=np.zeros(pixelseries.npx)
    totalcounts=np.zeros(pixelseries.npx)

    for i in np.arange(pixelseries.npx):
        counts=dataset[i,:]
        rvals[i], bvals[i], gvals[i], totalcounts[i] = spectorgb(pixelseries.energy, counts, red, green, blue)

    rgbimg, rvals, gvals, bvals = compile(rvals, gvals, bvals, xfmap.xres, pixelseries.nrows)

    export_show(rgbimg, rvals, gvals, bvals, dirs)

    
    return rgbimg, rvals, gvals, bvals
