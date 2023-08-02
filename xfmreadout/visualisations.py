import os
import logging
import numpy as np
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors

import xfmreadout.utils as utils
import xfmreadout.clustering as clustering
import xfmreadout.colours as colours


REDUCER=1

logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)

def plot_colour_embedding(embedding, palette_category_list, palette):

    #palette_category_list = np.arange(0,new_palette_embedding.shape[0])

    x=embedding.T[0]
    y=embedding.T[1]

    ### scatter plot with marginal axes
    sns.set_style('white')

    embed_plot = sns.jointplot(x=x, y=y,
                hue=palette_category_list, palette=palette,
                lw=0,
                joint_kws = dict(alpha=1.0),
                height=12, ratio=6
                )

    embed_plot.set_axis_labels('x', 'y', fontsize=16)

    embed_plot.ax_joint.legend_.remove()

    sns.despine(ax=None, left=True, bottom=True)
    fig = embed_plot.fig

    return fig




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



def show_map(data, dims, elements, target):
    """
        display a single map
    """
    img = utils.get_map(data, dims, elements, target)

    idx = utils.findelement(elements, target)

    print(f"ELEMENT MAP: {target}")
    print(f"({idx}), {target}, max: {np.max(img):.2f}, 98: {np.quantile(img,0.98):.2f}, avg: {np.average(img):.2f}")

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
        display categories as map image, with axes
    """

    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111)

    ncats=np.max(categories)+2

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    cmap = colors.ListedColormap(palette)

    catmap=utils.map_roll(categories+1,dims)

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.imshow(catmap, cmap=cmap)

    return fig


def category_map_direct( categories, dims, palette=None ):
    """
        display categories as whole map image
    """

    DPI=96

    fig = plt.figure(figsize=(dims[1]/DPI,dims[0]/DPI), dpi=DPI, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ncats=np.max(categories)+2

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    cmap = colors.ListedColormap(palette)

    catmap=utils.map_roll(categories+1,dims)

    print("creating direct category map")

    ax.imshow(catmap, cmap=cmap, aspect='auto')

    return fig



def table_classavg(classavg, elements):
    """
    display a table with class average values
    """
    concentration_averages = pd.DataFrame(data=classavg, columns=elements)
    
    print(tabulate(concentration_averages, headers='keys', tablefmt='psql'))

    return

def category_avgs(categories, elements, classavg, palette=None ):
    """
        display category spectra
    """
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    cmap = colors.ListedColormap(palette)

    n_clusters, category_list = utils.count_categories(categories)

    #ax.set_yscale('log')

    for i in range(n_clusters):
        icat=category_list[i]
        colour=cmap(i)
        ax.plot(elements, classavg[icat,:], linewidth=1, color=colour)

    fig.show()

    return

def category_boxplots(data, categories, elements):
    """
    display category:element boxplots
    """

    ncats=np.max(categories)+1
    catlist=range(ncats)

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
        palette=colours.build_palette(categories)

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

    embed_plot.set_axis_labels('x', 'y', fontsize=16)

    sns.despine(ax=None, left=True, bottom=True)
    fig = embed_plot.fig

    #plt.savefig('embedplot.png', transparent=True)
    #plt.show()

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


DPI=96

def contours_3d(kde):

    Z_local = np.copy(kde.Z)
    Z_local[Z_local < 0.00001] = -0.0005
    #Z_local = np.log(Z_local)

    #Make a 3D plot
    fig = plt.figure(figsize=(int(1500/DPI),int(800/DPI)))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(kde.X, kde.Y, Z_local,cmap='viridis',rstride=3,cstride=3,linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig

def plot_clusters(categories, classavg, embedding, kde, dims, output_directory=".", plot_kde=False, plot_margins=False):
    """
    display all plots for clusters
    """

    print("plotting") 

    if embedding.shape[1] == 2:
        print("using 2d embedding x") 
        #generate the palette from the categories, independent of distance
        palette=colours.build_aligned_palette(embedding, categories)
        embedding_2d = embedding
    else:
        #use the 3D embedding to colour the categories based on distance
        fig_embed_map = embedding_map(embedding, dims)
        fig_embed_map.savefig(os.path.join(output_directory,'embed_map.png'), transparent=False)  

        # produce 2D embedding for visualisation
        print("creating 2d embedding")
        ___, embedding_2d = clustering.reduce(embedding, "PCA", target_components=2) 
        palette=sns.color_palette(rgb_from_centroids(embedding, categories))

    if plot_margins:
        print("saving map with margins")        
        fig_cat_map = category_map(categories, dims, palette=palette)
        fig_cat_map.savefig(os.path.join(output_directory,'category_map.png'), transparent=False)    
    else:
        print("creating category map")
        fig_cat_map = category_map_direct(categories, dims, palette=palette)
        fig_cat_map.savefig(os.path.join(output_directory,'category_map.png'), transparent=False)  

    print("creating embedplot")    
    fig_embed = seaborn_embedplot(embedding_2d, categories, palette=palette)
    fig_embed.savefig(os.path.join(output_directory,'embeddings.png'), transparent=False)    
    
    if plot_kde and kde is not None:
        fig_contours = contours_3d(kde)
    
    #plt.show()

    return palette


def plot_classes(categories, labels, classavg, palette):
    """
    display details for categories
    
    """
    category_avgs(categories, labels, classavg, palette=palette)    