import numpy as np
import os
import matplotlib.pyplot as plt

cset = ['red', 'blue']

def predictdt(config, pxsum, dwell, timeconst):
    """
    predict missing deadtimes from countrate, dwell and time-constant
    must be calibrated for each time-constant, will fail if current TC is uncalibrated    
    """
    norm_sum=(pxsum/dwell)/2 #sum per pixel, corrected for dwell
                                            # corrected for no. detectors

    if timeconst == 0.5:    #hardcoded for now
        a=float(config['dtcalc_a'])
        c=float(config['dtcalc_c'])
        cutoff=float(config['dtcalc_cutoff'])

        dtpred=norm_sum*a+c
        dtpred=float(min(dtpred,cutoff))
    else:
        raise ValueError(f"Deadtime prediction not yet calibrated for TC={timeconst}")

    return dtpred

def postcalc(config, pixelseries, xfmap):
    timeconst = xfmap.timeconst
    dwell = xfmap.dwell
    ndet = pixelseries.ndet

    dtavg=np.sum(pixelseries.dt, axis=0)/ndet

    if len(pixelseries.flatsum) != len(dtavg):
        raise ValueError("sum and dt array sizes differ")

    dtpred=np.zeros(len(dtavg),dtype=np.float32)

    for i in range(len(pixelseries.flatsum)):
        dtpred[i]=predictdt(config, pixelseries.flatsum[i], dwell, timeconst)

    return dtpred, dtavg


def export(dir:str, dtpred, mergedsum):
    #NB: printing as if dtpred is list of lists -> newlines after each value
    #   not sure why, workaround is to pass as list of itself
    # https://stackoverflow.com/questions/42068144/numpy-savetxt-is-not-adding-comma-delimiter
    np.savetxt(os.path.join(dir, "pxstats_dtpred.txt"), [dtpred], fmt='%f', delimiter=",")
    np.savetxt(os.path.join(dir, "pxstats_mergedsum.txt"), [mergedsum], fmt='%d', delimiter=",")

    return


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
    
    #https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
    #seriously consider contoured plots
    #particularly 3rd answer by "Guilliame" using density_scatter
    
    return

def predhist(dt, dtpred, dir: str, ndet: int):
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)

    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("No. pixels")

    i=0
    labels = [ "measured", "predicted" ]

    for data in [ dt, dtpred ]:
        ax.hist(data, 100, fc=cset[i], alpha=0.5, label=f"{labels[i]}")
        i+=1

    fig.savefig(os.path.join(dir, 'predicted_deadtime_histograms.png'), dpi=150)
    return

def preddiffimage(dt, dtpred, dir: str, xres: int, yres: int, ndet: int):
       
    diffmap = dtpred-dt

    diffimage = diffmap.reshape(yres,xres)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    img = ax.imshow(diffimage, cmap='bwr')

    plt.colorbar(img, fraction=0.04, pad=0.04)

    plt.savefig(os.path.join(dir, 'predicted_difference_map.png'), dpi=150)
    return

def predscatter(dt, dtpred, sum, dir: str, ndet: int):
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(111)
    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("Counts")

    labels = [ "measured", "predicted" ]

    i=0
    for data in [ dt, dtpred ]:
        ax.scatter(sum, data, color=cset[i], marker='o', s=50, alpha=0.1, linewidths=None, label=f"{labels[i]}")
        i+=1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Detector:")
    #NB: works but not sure why... box appears to right
    #from: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    fig.savefig(os.path.join(dir, 'predicted_deadtime_scatter.png'), dpi=150)
    
    return


def dtplots(config, dir: str, dt, sum, dtpred, dtavg, mergedsum, xres: int, yres: int, ndet: int):

    dthist(dt, dir, ndet)
    dtimages(dt, dir, xres, yres, ndet)
    
    if config['PARSEMAP'] and (np.amax(sum) > 0):
        
        if ndet == 2:
        #difference map requires two detectors
            diffimage(sum, dir, xres, yres, ndet)
        
        dtscatter(dt, sum, dir, ndet)
    elif (np.amax(sum) <= 0):
        raise ValueError("Sum array is empty or zero - cannot generate sum plots")

    predhist(dtavg, dtpred, dir, ndet)
    preddiffimage(dtavg, dtpred, dir, xres, yres, ndet)
    predscatter(dtavg, dtpred, mergedsum, dir, ndet)

    plt.close()

    return 
