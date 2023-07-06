import copy
import os
import logging
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from matplotlib import colors
from PIL import Image

import xfmreadout.utils as utils
import xfmreadout.clustering as clustering


REDUCER=1

logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)


def shuffle_palette(palette):
    """
    shuffles palette into N blocks of spaced sequential colours
        breaks up a sequential colour sequence
    """

    NREPEATS=3

    mod = int(np.floor(len(palette)/(len(palette)/NREPEATS)))

    newpalette=copy.deepcopy(palette)

    for j in range(mod):
        for i in range(len(palette)):
            if i%mod == j:
                newpalette.append(palette[i])

    del newpalette[0:len(palette)]

    return newpalette

def build_palette(categories,cmapname=cc.glasbey_light,shuffle=False):
    """
    generates a categorical palette based on size
        includes sequential shuffling for large category nos. 
        applies grey to unassigned/negative
    """

    GREY=( 0.5, 0.5, 0.5 )

    cat_min=np.min(categories)
    cat_max=np.max(categories)
    num_cats=cat_max-cat_min+1

    if num_cats <= 10:
        cmapname="deep"
        shuffle=False
    elif num_cats <=12:
        cmapname="Set3"
        shuffle=False

    if cat_min < 0:
        palette=sns.color_palette(cmapname,num_cats-1)

        if shuffle == True:
            palette=shuffle_palette(palette)

        palette.insert( 0, GREY )

    elif cat_min == 0:
        palette=sns.color_palette(cmapname,num_cats)

        if shuffle == True:
            palette=shuffle_palette(palette)
    else:
        raise ValueError(f"minimum category {cat_min} > 0")

    return palette


def show_map(data, dims, elements, target):
    """
        display a single map
    """
    img = utils.get_map(data, dims, elements, target)

    print(f"ELEMENT MAP: {target}")
    print(f"{target}, max: {np.max(img):.2f}, 98: {np.quantile(img,0.98):.2f}, avg: {np.average(img):.2f}")

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(111)

    display = ax.imshow(img, cmap='plasma')

    plt.show()

    return


def tricolour(r, g, b):
    """
    display a 3-colour RGB, normalising each channel
    """
    r = utils.norm_channel(r)
    g = utils.norm_channel(g)
    b = utils.norm_channel(b)

    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111)

    rgb = np.stack((r,g,b), axis=2)

    ax.imshow(rgb)    

    return fig

def tricolour_enames(e1:str, e2:str, e3:str, data, dims, elements):
    """
    display a 3-colour RGB from element names
    normalise each channel
    """
    r = utils.get_map(data, dims, elements, e1)
    g = utils.get_map(data, dims, elements, e2)
    b = utils.get_map(data, dims, elements, e3)    

    fig = tricolour(r, g, b)

    return fig

def embedding_map(embedding, dims):
    """
    display an RGB map coloured by embedding values in each dimension
    
    visualises spectral distance between points
    """
    embedding_map = utils.map_roll(embedding, dims)

    fig = tricolour(embedding_map[:,:,0], embedding_map[:,:,1], embedding_map[:,:,2])

    return fig

def category_map ( categories, dims, palette=None ):
    """
        display categories as map image
    """

    #KCMAPS=["tab10"]    #colourmaps for kmeans

    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111)

    ncats=np.max(categories)+2
    print(ncats)

    #axcm=cm.get_cmap(KCMAPS[0], ncats)

    #cmap=axcm(range(ncats))
    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=build_palette(categories)

    cmap = colors.ListedColormap(palette)

    catmap=utils.map_roll(categories+1,dims)

    print(np.min(categories))
    print(np.min(catmap))

    ax.tick_params(axis='both', which='major', labelsize=16)

    #show this category image
    ax.imshow(catmap, cmap=cmap)

    return fig

    #fig.savefig(os.path.join(EMBED_DIR,"cluster_map.png"), dpi=200)

def category_avgs(categories, elements, classavg, palette=None ):
    """
        display category spectra
    """
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=build_palette(categories)

    cmap = colors.ListedColormap(palette)

    ncats=np.max(categories)

    elids=range(len(elements))

    #ax.set_yscale('log')

    for i in range(ncats-1):
        ax.plot(elements, classavg[i,:], linewidth=1)


    fig.show()

    return

def category_boxplots(data, categories, elements):
    """
    display category:element boxplots
    """

    ncats=np.max(categories)+1
    catlist=range(ncats)

    print(categories.shape)
    print(data.shape)

    boxdata=np.zeros((ncats,len(elements),categories.shape[1]))


    #fig = plt.figure(figsize =(12, 6))

    fig, (ax) = plt.subplots(ncats, 1, figsize=(12, 24))

    #ax = fig.add_subplot(111)

    for cat_idx in catlist:
        assigned=np.where(categories == cat_idx, True, False)

        selected=[]

        for el_idx, ename in enumerate(elements):
            boxdata[cat_idx,el_idx,:]=data[:,el_idx]*assigned
            selected.append(data[assigned,el_idx])
        
            # Creating plot

        #ax[cat_idx].set_yscale('log')

        ax[cat_idx].boxplot(selected, labels=elements, whis=[0.01,99.99])

    fig.show()

    return fig

def seaborn_embedplot(embedding, categories, palette=None):
    """
    display seaborn plot of embedding space
    """

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=build_palette(categories)

    x=embedding.T[0]
    y=embedding.T[1]

    ### scatter plot with marginal axes
    sns.set_style('white')

    embed_plot = sns.jointplot(x=x, y=y,
                hue=categories, palette=palette,
                lw=0,
                joint_kws = dict(alpha=0.01),
                height=12, ratio=6
                )

    #xlim=[-3,3], ylim=[-3,3],

    sns.despine(ax=None, left=True, bottom=True)
    fig = embed_plot.fig

    #plt.savefig('embedplot.png', transparent=True)
    plt.show()

    return fig

def seaborn_kdeplot(embedding, categories):
    """
    display  plot of embedding space as kernel density
    
    filled colors, transparent background

    """
    x=embedding.T[0]
    y=embedding.T[1]

    sns.set_style('white')
    kdeplot = sns.kdeplot(x=x, y=y,
                hue=categories,
                fill=True,
                legend=False)
#    ax.ax_marg_x.remove()
#    ax.ax_marg_y.remove()
    
    fig = kdeplot.fig

    #ax = sns.despine(ax=None, left=True, bottom=True)

    plt.show()

    return fig

def seaborn_kdecontours(embedding, categories):
    """
    display  plot of embedding space as contours
    
    """
    x=embedding.T[0]
    y=embedding.T[1]

    sns.set_style('white')
    ax = sns.jointplot(x=x, y=y,
                cut = 0, hue=categories,
                palette=sns.color_palette("dark"),
                kind='kde', fill=False,
                height=15, ratio=6,
                joint_kws = dict(alpha=0.4),
                marginal_kws=dict(fill=True),
                legend=False)
    ax.ax_marg_x.remove()
    ax.ax_marg_y.remove()
    ax = sns.despine(ax=None, left=True, bottom=True)
    #plt.savefig('kde_tr_fill.png', transparent=True)
    plt.show()

def rgb_from_centroids(embedding, categories):
    """
    create RGB indexes based on centroids of each cluster
    """

    centroids = utils.compile_centroids(embedding, categories)

    centroids_rgb = np.zeros(centroids.shape, dtype=np.float32)

    for i in range(centroids.shape[1]):
        centroids_rgb[:,i] = utils.norm_channel_float(centroids[:,i],new_max=1.0)

    centroids_rgb[0] = (0.5, 0.5, 0.5)

    return centroids_rgb

    #cmap = LinearSegmentedColormap.from_list('custom', centroids_rgb, N=centroids.shape[0])


def plot_clusters(categories, classavg, embedding, dims, output_directory="."):
    """
    display all plots for clusters
    
    """

    if embedding.shape[1] == 2:
        #generate the palette from the categories, independent of distance
        palette=build_palette(categories)
    else:
        #use the 3D embedding to colour the categories based on distance
        fig_embed_map = embedding_map(embedding, dims)
        fig_embed_map.savefig(os.path.join(output_directory,'embed_map.png'), transparent=False)  

        # produce 2D embedding for visualisation
        ___, embedding_2d = clustering.reduce(embedding, "PCA", target_components=2) 
        colour_array=rgb_from_centroids(embedding, categories)
        palette=sns.color_palette(colour_array)

    fig_cat_map = category_map(categories, dims, palette=palette)
    fig_cat_map.savefig(os.path.join(output_directory,'category_map.png'), transparent=False)    

    fig_embed = seaborn_embedplot(embedding_2d, categories, palette=palette)
    fig_embed.savefig(os.path.join(output_directory,'embeddings.png'), transparent=False)    

    return palette


def plot_classes(categories, labels, classavg, palette):
    """
    display details for categories
    
    """
    category_avgs(categories, labels, classavg, palette=palette)    