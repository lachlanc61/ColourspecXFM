import sys
import numpy as np
import scipy as sp
import pybaselines.smooth
import matplotlib.pyplot as plt

#-----------------------------------
#MODIFIABLE CONSTANTS
#-----------------------------------

SNIPWINDOW=50   #width-window for SNIP algorithm - 50 is default
LOWCUT=80       #low cut point for SNIP

YIELD_LINES=np.array([ 0.5  ,  1.   ,  1.486,  1.739,  2.013,  2.304,  2.957,  3.312, \
        3.69 ,  4.508,  5.411,  5.894,  6.398,  6.924,  7.471,  8.04 , \
       10.53 , 11.907, 13.373, 14.14 , 15.744, 17.441, 25.   ])


YIELD_FACTORS=np.array([5.00000e-04, 3.00000e-05, 1.20000e-05, 1.75000e-05, 2.07492e-05, \
       3.66851e-05, 4.05230e-05, 5.59011e-05, 6.93785e-05, 1.75159e-04, \
       3.61108e-04, 4.65544e-04, 6.04780e-04, 6.97253e-04, 5.42119e-04, \
       5.58549e-04, 5.01282e-04, 4.43092e-04, 3.77418e-04, 3.68455e-04, \
       2.00000e-04, 1.50000e-04, 1.00000e-04])


#-----------------------------------
#INITIALISE
#-----------------------------------

this = sys.modules[__name__]

#-----------------------------------
#FUNCTIONS
#-----------------------------------


def plotspline(CORRECTION_FACTORS, energy, yield_lines, yield_inverted):
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('energy (kV)') 
    ax1.set_ylabel('Intensity') 
    ax1.axis(xmin=0,xmax=30)

    ax1.set_yscale('log')
    #ax1.axis(ymin=0.00005,ymax=1)

    ax1.plot(energy, CORRECTION_FACTORS, "r+" )

    ax1.plot(yield_lines, yield_inverted, "ko")

    plt.show()


def plotcorrection(spectrum, energy, bg, sub, adj):

    #return adj

    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('energy (kV)') 
    ax1.set_ylabel('Intensity') 
    ax1.axis(xmin=0,xmax=30)

    #ax1.set_yscale('log')
    #ax1.axis(ymin=0.8,ymax=10000000)

    ax1.plot(energy, spectrum, color="grey")
    ax1.plot(energy, bg, color="red")
    ax1.plot(energy, sub, color="blue")
    ax1.plot(energy, adj, color="green")



def initialise(energy):

    yield_inverted=1/YIELD_FACTORS

    spline = sp.interpolate.UnivariateSpline(YIELD_LINES, yield_inverted, k=1)

    CORRECTION_FACTORS = spline(energy)

    if False:
        plotspline(CORRECTION_FACTORS, energy, yield_lines, yield_inverted)

    return CORRECTION_FACTORS


def correct_spec(data, CORRECTION_FACTORS):
    spectrum=data
    spectrum[spectrum<1]=1

    bg = pybaselines.smooth.snip(spectrum, 30, decreasing=True, smooth_half_window=1)[0]
    bg = bg.astype(np.uint32)
    #WARNING: bug here, some 1.0s become 0s
    #   not sure if issue with pybaselines or something else
    #   bg is initially a np array but does not report a dtype - issue?
    bg[bg < 1] = 1
    sub = spectrum-bg

    #set all values <1 to 0 so they don't get scaled by the multiplier
    #sub[sub < 1] = 0

    adj=sub*CORRECTION_FACTORS
    adj=adj.astype(np.uint32)

    if False:
        plotcorrection(spectrum, energy, bg, sub, adj)      #energy not in local namespace otherwise

    return adj    


def calc_corrected(dataset, energy, npx, nchan):

    print("fitting baselines")

    CORRECTION_FACTORS = initialise(energy)

    corrected=np.zeros((npx,nchan),dtype=np.uint32)

    for i in np.arange(npx):
        corrected[i]=correct_spec(dataset[i], CORRECTION_FACTORS)

    return corrected