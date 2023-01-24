import numpy as np
import codecs
import os
import matplotlib.pyplot as plt

cset = ['red', 'blue']

def dthist(dt, odir, ndet):
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)

    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("No. pixels")

    for i in np.arange(ndet):
        ax.hist(dt[i], 100, fc=cset[i], alpha=0.5, label=f"{i}")

    ax.legend(loc=1, title="Detector:")

    fig.savefig(os.path.join(odir, 'deadtime_histograms.png'), dpi=150)
    fig.show()
    return

def dtimages(dt, odir, xres, yres, ndet):
    fig = plt.figure(figsize=(8,4))

    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    ax0.set_title("Detector: 0")
    ax0.tick_params(axis='x',colors='red')
    ax0.tick_params(axis='y',colors='red')
    for spine in ax0.spines.values():
        spine.set_linewidth(2)
        spine.set_color('red')

    ax1.set_title("Detector: 1")
    ax1.tick_params(axis='x',colors='blue')
    ax1.tick_params(axis='y',colors='blue')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_color('blue')

    dtimage0 = dt[0].reshape(yres,xres)
    dtimage1 = dt[1].reshape(yres,xres)

    ax0.imshow(dtimage0, cmap="magma")
    ax1.imshow(dtimage1, cmap="magma")

    fig.savefig(os.path.join(odir, 'deadtime_maps.png'), dpi=150)
    fig.show()
    return

def diffimage(sum, odir, xres, yres, ndet):
    diffmap = sum[0]-sum[1]

    diffimage = diffmap.reshape(yres,xres)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    img = ax.imshow(diffimage, cmap='bwr')

    plt.colorbar(img, fraction=0.04346, pad=0.04)

    plt.savefig(os.path.join(odir, 'difference_map.png'), dpi=150)
    plt.show()
    return

def dtscatter(dt, sum, odir, ndet):
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(111)
    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("Counts")

    ax.scatter(dt[0],sum[0], color="red", marker='o', s=50, alpha=0.1, linewidths=None, label="0")
    ax.scatter(dt[1],sum[0], color="blue", marker='o', s=50, alpha=0.1, linewidths=None, label="1")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Detector:")
    #NB: works but not sure why... box appears to right
    #from: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    fig.savefig(os.path.join(odir, 'deadtime_vs_counts.png'), dpi=150)
    fig.show()

def dtplots(config, odir, dt, sum, xres, yres, ndet):

    dthist(dt, odir, ndet)
    dtimages(dt, odir, xres, yres, ndet)
    
    if config['PARSEMAP'] and (np.amax(sum) > 0):
        diffimage(sum, odir, xres, yres, ndet)
        dtscatter(dt, sum, odir, ndet)
    elif (np.amax(sum) <= 0):
        raise ValueError("Sum array zero/not present, cannot generate sum plots")

    return 