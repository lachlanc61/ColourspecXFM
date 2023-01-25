import numpy as np
import os
import matplotlib.pyplot as plt

cset = ['red', 'blue']

def predictdt(config, pxsum, dwell, timeconst):
    """
    predict missing deadtimes from countrate, dwell and time-constant
    must be calibrated for each time-constant, will fail if current TC is uncalibrated    
    """
    norm_sum=int(round(pxsum/dwell), 0)

    if timeconst == 0.5:    #hardcoded for now
        a=float(config['dtcalc_a'])
        c=float(config['dtcalc_c'])
        cutoff=float(config['dtcalc_cutoff'])

        dtpred=norm_sum*a+c
        dtpred=float(round(min(dtpred,cutoff),2))
    else:
        raise ValueError(f"Deadtime prediction not calibrated for TC={timeconst}")

def dthist(dt, dir: str, ndet: int):
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)

    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("No. pixels")

    for i in np.arange(ndet):
        ax.hist(dt[i], 100, fc=cset[i], alpha=0.5, label=f"{i}")

    ax.legend(loc=1, title="Detector:")

    fig.savefig(os.path.join(dir, 'deadtime_histograms.png'), dpi=150)
    return

def dtimages(dt, dir: str, xres: int, yres: int, ndet: int):
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

        dtimage = dt[i].reshape(yres,xres)

        ax[0,i].imshow(dtimage, cmap="magma")

    fig.savefig(os.path.join(dir, 'deadtime_maps.png'), dpi=150)
    return

def diffimage(sum, dir: str, xres: int, yres: int, ndet: int):
    
    if ndet != 2:
        raise ValueError("Number of detectors != 2, difference map not possible")
       
    sum=sum.astype(float)
         
    diffmap = sum[0]-sum[1]

    diffimage = diffmap.reshape(yres,xres)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    img = ax.imshow(diffimage, cmap='bwr')

    plt.colorbar(img, fraction=0.04346, pad=0.04)

    plt.savefig(os.path.join(dir, 'difference_map.png'), dpi=150)
    return

def dtscatter(dt, sum, dir: str, ndet: int):
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(111)
    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("Counts")

    for i in np.arange(ndet):
        ax.scatter(dt[i],sum[i], color=cset[i], marker='o', s=50, alpha=0.1, linewidths=None, label=f"{i}")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Detector:")
    #NB: works but not sure why... box appears to right
    #from: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    fig.savefig(os.path.join(dir, 'deadtime_vs_counts.png'), dpi=150)
    return

def dtplots(config, dir: str, dt, sum, xres: int, yres: int, ndet: int):

    dthist(dt, dir, ndet)
    dtimages(dt, dir, xres, yres, ndet)
    
    if config['PARSEMAP'] and (np.amax(sum) > 0):
        
        if ndet == 2:
        #difference map requires two detectors
            diffimage(sum, dir, xres, yres, ndet)
        
        dtscatter(dt, sum, dir, ndet)
    elif (np.amax(sum) <= 0):
        raise ValueError("Sum array is empty or zero - cannot generate sum plots")

    plt.close()

    return 