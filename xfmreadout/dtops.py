import numpy as np
import os
import matplotlib.pyplot as plt

cset = ['red', 'blue']

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

def predict_dt(config, pixelseries, xfmap):
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

    if len(pixelseries.flatsum) != len(pixelseries.dtflat):
        raise ValueError("sum and dt array sizes differ")

    dtmod=np.zeros((len(pixelseries.dtflat),ndet),dtype=np.float32) 

    if timeconst == 0.5:    #currently hardcoded to TC = 0.5 us
        for i in range(len(pixelseries.flatsum)):
            for j in range(ndet):
                dtmod[i,j]=predict_single_dt(config, pixelseries.flatsum[i], dwell, timeconst)
    else:
        raise ValueError(f"Deadtime prediction not yet calibrated for TC={timeconst}")        

    return dtmod


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
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)

    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("No. pixels")

    for i in np.arange(ndet):
        ax.hist(dt[:,i], 100, fc=cset[i], alpha=0.5, label=f"{i}")

    ax.legend(loc=1, title="Detector:")

    fig.savefig(os.path.join(dir, 'deadtime_histograms.png'), dpi=150)
    return

def dtimages(dt, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the deadtimes as a map image
    """
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
    return

def diffimage(sum, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the differences in deadtimes between detectors as a map image
    """    
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
    return

def dtscatter(dt, sum, dir: str, ndet: int):
    """
    produce scatterplot of deadtime vs counts per pixel
    """  
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
    
    #https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
    #seriously consider contoured plots
    #particularly 3rd answer by "Guilliame" using density_scatter
    
    return

def predhist(dt, dtmod, dir: str, ndet: int):
    """
    generate the predicted deadtime histogram plot
    """ 
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)

    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("No. pixels")

    i=0
    labels = [ "measured", "predicted" ]

    for data in [ dt, dtmod ]:
        ax.hist(data, 100, fc=cset[i], alpha=0.5, label=f"{labels[i]}")
        i+=1

    fig.savefig(os.path.join(dir, 'predicted_deadtime_histograms.png'), dpi=150)
    return

def preddiffimage(dt, dtmod, dir: str, xres: int, yres: int, ndet: int):
    """
    plot the differences in predicted deadtimes between detectors as a map image
    """          
    diffmap = dtmod-dt

    diffimage = diffmap.reshape(yres,xres)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    img = ax.imshow(diffimage, cmap='bwr')

    plt.colorbar(img, fraction=0.04, pad=0.04)

    plt.savefig(os.path.join(dir, 'predicted_difference_map.png'), dpi=150)
    return

def predscatter(dt, dtmod, sum, dir: str, ndet: int):
    """
    produce scatterplot of predicted deadtime vs counts per pixel
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
    
    return


def dtplots(config, dir: str, dt, sum, dtmod, dtavg, mergedsum, xres: int, yres: int, ndet: int, INDEX_ONLY: bool):
    """
    produce all deadtime-related plots
    """

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

    #predhist(dtavg, dtmod, dir, ndet)
    #preddiffimage(dtavg, dtmod, dir, xres, yres, ndet)
    #predscatter(dtavg, dtmod, mergedsum, dir, ndet)

    plt.close()

    return 
