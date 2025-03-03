import numpy as np
import os
import time
import matplotlib.pyplot as plt
from numpy.polynomial  import Polynomial

import logging
logger = logging.getLogger(__name__)

cset = ['red', 'pink', 'blue', 'lightblue']

def dt_stats(dt):
    """
    print deadtime statistics to stdout
    """
    print(
        "---------------------------\n"
        "DT STATISTICS\n"
        "---------------------------"
    )
    
    for i in range(dt.shape[1]):
        dt_mean = np.mean(dt[:,i])    
        print(f"detector {i} mean: {dt_mean: .2f}")
    
    dt_mean = np.mean(dt)  
    print(f"overall mean: {dt_mean: .2f}")
    print("---------------------------")
    return dt_mean


def predict_single_dt(config, pxsum, dwell, timeconst):
    """
    predict missing deadtimes from countrate, dwell and time-constant
    must be calibrated for each time-constant, will fail if current TC is uncalibrated    
    """
    norm_sum=(pxsum/dwell)/2 #sum per pixel, corrected for dwell
                                            # corrected for no. detectors

    a=float(config['dtcalc_a'])
    c=float(config['dtcalc_c'])
    cutoff=float(config['dtcalc_cutoff'])

    dtmod=norm_sum*a+c
    dtmod=float(min(dtmod,cutoff))

    return dtmod

def predict_dt_flat(config, pixelseries, xfmap):
    """
    predict deadtimes-per-pixel from counts
    
    """

    #FUTURE: alternate behaviour if flag:
    #           fill deadtimes with fixed value from average
    timeconst = xfmap.timeconst
    dwell = xfmap.dwell
    ndet = pixelseries.ndet

    if pixelseries.parsed == False and not np.max(pixelseries.flatsum) > 0:
        raise ValueError("Deadtime prediction requires parsed map with flattened data")

    if len(pixelseries.flatsum) != len(pixelseries.dt[:,0]):
        raise ValueError("sum and dt array sizes differ")

    dtmod=np.zeros((len(pixelseries.dt[:,0]),ndet),dtype=np.float32)

    if timeconst == 0.5:    #currently hardcoded to TC = 0.5 us
        for i in range(len(pixelseries.flatsum)):
            for j in range(ndet):
                dtmod[i,j]=predict_single_dt(config, pixelseries.flatsum[i], dwell, timeconst)
    else:
        raise ValueError(f"Deadtime prediction not yet calibrated for TC={timeconst}")        

    return dtmod

def dt_poly3(summed_data, dwell: float):
    """
    predict deadtimes from per-pixel-per-det counts
    """
    # preset coefficients for polynomial
    #   calculated from series of geological standards with varied deadtimes
    COEFFICIENTS = [0.23438781, 0.21464015, 0.05402071, 0.02689396]
    DOMAIN = [ 4.08008563, 99.727441  ]
    WINDOW = [-1.,  1.]

    SATURATION_CUTOFF = 65
    MAX_CUTOFF = 80

    # Create the polynomial
    q = Polynomial(COEFFICIENTS, domain=DOMAIN, window=WINDOW)

    predicted = q(summed_data/dwell)*100

    lower_mask = np.where(predicted > SATURATION_CUTOFF)

    if len(lower_mask[0]) > 0:
        print(f"WARNING: When predicting deadtimes, {len(lower_mask[0])} pixels in saturation zone ({len(lower_mask[0])/summed_data.shape[0]*100:.2f}% of pixels at >{SATURATION_CUTOFF}% DT)")

    upper_mask = np.where(predicted > MAX_CUTOFF)

    if len(upper_mask[0]) > 0:
        print(f"WARNING: When predicting deadtimes, {len(upper_mask[0])} pixels at >{MAX_CUTOFF}% deadtime; normalised to 80%")

    return predicted


def predict_dt(pixelseries, xfmap):
    """
    predict deadtimes-per-pixel from per-detector counts
    
    """

    #FUTURE: alternate behaviour if flag:
    #           fill deadtimes with fixed value from average
    timeconst = xfmap.timeconst
    dwell = xfmap.dwell
    n_det = pixelseries.ndet

    if pixelseries.parsed == False and not np.max(pixelseries.sum) > 0:
        raise ValueError("Deadtime prediction requires sum-per-pixel-per-detector")

    if len(pixelseries.sum) != len(pixelseries.dt):
        raise ValueError("sum and dt array sizes differ")

    dt_pred=np.zeros(pixelseries.dt.shape,dtype=np.float32)

    if timeconst == 0.5:    #currently hardcoded to TC = 0.5 us
        dt_pred = dt_poly3(pixelseries.sum, dwell)
    else:
        raise ValueError(f"Deadtime prediction not yet calibrated for TC={timeconst}")   
         
    return dt_pred


def export(dir:str, dtmod, flatsum):
    """
    export the derived deadtime values
    - pass if arrays are zero - eg. if map was not parsed to generate them
    """

    #NB: printing as if dtmod is list of lists -> newlines after each value
    #   not sure why, workaround is to pass as list of itself
    # https://stackoverflow.com/questions/42068144/numpy-savetxt-is-not-adding-comma-delimiter
    if np.sum(dtmod[0] > 0) and np.sum(flatsum > 0):
        np.savetxt(os.path.join(dir, "pxstats_dtmod.txt"), [dtmod], fmt='%f', delimiter=",")
        np.savetxt(os.path.join(dir, "pxstats_flatsum.txt"), [flatsum], fmt='%d', delimiter=",")
    else:
        pass
    return


def dthist(dt, dir: str, ndet: int):
    """
    generate the deadtime histogram plot
    """
    try:
        fig = plt.figure(figsize=(6,4))

        ax = fig.add_subplot(111)

        ax.set_xlabel("Deadtime (%)")
        ax.set_ylabel("No. pixels")

        for i in np.arange(ndet):
            ax.hist(dt[:,i], 100, fc=cset[i], alpha=0.5, label=f"{i}")

        ax.legend(loc=1, title="Detector:")

        fig.savefig(os.path.join(dir, 'deadtime_histograms.png'), dpi=150)
        time.sleep(2)
        plt.clf()    
    except:
        print("WARNING: could not complete dt histogram plot")
    return

def dtimages(dt, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the deadtimes as a map image
    """
    try:
        #https://stackoverflow.com/questions/52273546/matplotlib-typeerror-axessubplot-object-is-not-subscriptable
        #squeeze kwarg forces 1x1 plot to behave as a 2D array so subscripting works
        fig, ax = plt.subplots(1, ndet, figsize=(8,4), squeeze=False)

        cset = ['red', 'blue']

        for i in np.arange(ndet):
            #ax now a 2D array because of squeeze - no sure if this index is correct for multiple plots, may be other axis
            ax[0,i].set_title(f"Detector: {i}")
            ax[0,i].tick_params(axis='x',colors=cset[i])
            ax[0,i].tick_params(axis='y',colors=cset[i])
            for spine in ax[0,i].spines.values():
                spine.set_linewidth(2)
                spine.set_color(cset[i])

            dtimage = dt[:,i].reshape(yres,xres)

            ax[0,i].imshow(dtimage, cmap="magma")

        fig.savefig(os.path.join(dir, 'deadtime_maps.png'), dpi=150)
        time.sleep(2)
        plt.clf()    
    except:
        print("WARNING: could not complete dt image output")
    return

def diffimage(sum, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the differences in deadtimes between detectors as a map image
    """    
    try:
        if ndet != 2:
            raise ValueError("Number of detectors != 2, difference map not possible")
        
        sum=sum.astype(float)
            
        diffmap = sum[:,0]-sum[:,1]

        diffimage = diffmap.reshape(yres,xres)

        fig = plt.figure(figsize=(6,6))

        ax = fig.add_subplot(111)

        img = ax.imshow(diffimage, cmap='bwr')

        plt.colorbar(img, fraction=0.04346, pad=0.04)

        plt.savefig(os.path.join(dir, 'difference_map.png'), dpi=150)
        time.sleep(2)
        plt.clf()   
    except:
        print("WARNING: could not complete dt difference image")
    return

def dtscatter(dt, sum, dir: str, ndet: int):
    """
    produce scatterplot of deadtime vs counts per pixel
    """  
    try:
        fig = plt.figure(figsize=(8,4))

        ax = fig.add_subplot(111)
        ax.set_xlabel("Deadtime (%)")
        ax.set_ylabel("Counts")

        for i in np.arange(ndet):
            ax.scatter(dt[:,i],sum[:,i], color=cset[i], marker='o', s=50, alpha=0.1, linewidths=None, label=f"{i}")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Detector:")
        #NB: works but not sure why... box appears to right
        #from: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

        fig.savefig(os.path.join(dir, 'deadtime_vs_counts.png'), dpi=150)
        time.sleep(2)
        plt.clf()
        #https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
        #seriously consider contoured plots
        #particularly 3rd answer by "Guilliame" using density_scatter
    except:
        print("WARNING: could not complete dt scatter plot")
    return

def predhist(dt, dtmod, dir: str, ndet: int):
    """
    generate the predicted deadtime histogram plot
    """ 
    try:
        fig = plt.figure(figsize=(6,4))

        ax = fig.add_subplot(111)

        ax.set_xlabel("Deadtime (%)")
        ax.set_ylabel("No. pixels")

        i=0
        labels = [ "measured", "predicted" ]

        for data in [ dt, dtmod ]:
            for det in range(ndet):
                ax.hist(data[:,det], 100, fc=cset[i*2+det], alpha=0.5, label=f"{labels[i]}, {det}")
            i+=1

        fig.savefig(os.path.join(dir, 'predicted_deadtime_histograms.png'), dpi=150)
        time.sleep(2)
        plt.clf()
    except:
        print("WARNING: could not complete predicted dt histogram plot")    
    return

def preddiffimage(dt, dtmod, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the differences in predicted deadtimes between detectors as a map image

    DEPRECATED
    """          
    diffmap = dtmod-dt

    diffimage = diffmap.reshape(yres,xres)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    img = ax.imshow(diffimage, cmap='bwr')

    plt.colorbar(img, fraction=0.04, pad=0.04)

    plt.savefig(os.path.join(dir, 'predicted_difference_map.png'), dpi=150)
    time.sleep(2)
    plt.clf()
    return

def predscatter(dt, dtmod, sum, dir: str, ndet: int):
    """
    produce scatterplot of predicted deadtime vs counts per pixel

    DEPRECATED
    """  
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(111)
    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("Counts")

    labels = [ "measured", "predicted" ]

    i=0
    for data in [ dt, dtmod ]:
        ax.scatter(sum, data, color=cset[i], marker='o', s=50, alpha=0.1, linewidths=None, label=f"{labels[i]}")
        i+=1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Detector:")
    #NB: works but not sure why... box appears to right
    #from: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    fig.savefig(os.path.join(dir, 'predicted_deadtime_scatter.png'), dpi=150)
    plt.clf()
    time.sleep(2)
    return


def dtplots(config, dir: str, dt, sum, dtmod, xres: int, yres: int, ndet: int, INDEX_ONLY: bool):
    """
    produce all deadtime-related plots
    """
    try:
        dthist(dt, dir, ndet)
        dtimages(dt, dir, xres, yres, ndet)
        diffimage(sum, dir, xres, yres, ndet)    
        dtscatter(dt, sum, dir, ndet)    
        
        if not INDEX_ONLY and (np.amax(sum) > 0):
            
            if ndet == 2:
            #difference map requires two detectors
                diffimage(sum, dir, xres, yres, ndet)
            
            dtscatter(dt, sum, dir, ndet)
        elif (np.amax(sum) <= 0):
            raise ValueError("Sum array is empty or zero - cannot generate sum plots")

        if np.max(dtmod) > 0 and dtmod.shape[1] == 2:
            #predicted deadtime map requieres active prediction with two detectors
            predhist(dt, dtmod, dir, ndet)   
        else:
            pass
    except:
        print("WARNING: could not complete dt plots")
    
    finally:
        plt.close()
        
        return 
