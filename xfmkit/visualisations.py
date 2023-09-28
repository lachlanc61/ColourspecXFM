import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors

import xfmkit.utils as utils
import xfmkit.clustering as clustering
import xfmkit.colours as colours
import xfmkit.tabular as tabular

import logging
logger = logging.getLogger(__name__)

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

    FIRST_CATEGORISED=1

    centroids = utils.compile_centroids(embedding, categories)

    centroids_rgb = np.zeros(centroids.shape, dtype=np.float32)

    for i in range(FIRST_CATEGORISED, centroids.shape[1]):
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
    r_ = utils.norm_channel(r)
    g_ = utils.norm_channel(g)
    b_ = utils.norm_channel(b)

    print(f"R, G, B max: {np.max(r_)}, {np.max(g_)}, {np.max(b_)}")

    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111)

    rgb = np.stack((r_,g_,b_), axis=2)

    ax.imshow(rgb)    

    return fig


def tricolour_pixelset(e_red:str, e_green:str, e_blue:str, pxs):
    """
    display a 3-colour RGB from element names and a pixelset object
    normalise each channel
    """
    ridx = utils.findelement(pxs.labels, e_red)
    gidx = utils.findelement(pxs.labels, e_green)
    bidx = utils.findelement(pxs.labels, e_blue)   

    r = pxs.data.mapview[:,:,ridx]
    g = pxs.data.mapview[:,:,gidx]
    b = pxs.data.mapview[:,:,bidx]   

    print(f"R, G, B max: {np.max(r)}, {np.max(g)}, {np.max(b)}")

    fig = tricolour(r, g, b)

    return fig

def tricolour_explicit(e_red:str, e_green:str, e_blue:str, data, dims, labels):
    """
    display a 3-colour RGB from element names
    normalise each channel
    """
    ridx = utils.findelement(labels, e_red)
    gidx = utils.findelement(labels, e_green)   
    bidx = utils.findelement(labels, e_blue)   


    r = data.mapview[:,:,ridx]
    g = data.mapview[:,:,gidx]
    b = data.mapview[:,:,bidx]   



    fig = tricolour(r, g, b)

    return fig


"""
DEPRECATE
def tricolour_enames(e1:str, e2:str, e3:str, data, dims, elements):
    ""
    display a 3-colour RGB from element names
    normalise each channel
    ""
    r = utils.get_map(data, dims, elements, e1)
    g = utils.get_map(data, dims, elements, e2)
    b = utils.get_map(data, dims, elements, e3)    

    fig = tricolour(r, g, b)

    return fig
"""


def embedding_map(embedding, dims):
    """
    display an RGB map coloured by embedding values in each dimension
    
    visualises spectral distance between points
    """
    embedding_map = utils.map_roll(embedding, dims)

    fig = tricolour(embedding_map[:,:,0], embedding_map[:,:,1], embedding_map[:,:,2])

    return fig

def category_map (categories, dims, palette=None ):
    """
        display categories as map image, with axes
    """

    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111)

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    cmap = colors.ListedColormap(palette)

    catmap=utils.map_roll(categories,dims)

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

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    cmap = colors.ListedColormap(palette)

    catmap=utils.map_roll(categories,dims)

    ax.imshow(catmap, cmap=cmap, aspect='auto')

    return fig

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

def seaborn_embedplot(embedding, categories, palette=None, labels=[]):
    """
    display seaborn plot of embedding space
    """

    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=colours.build_palette(categories)

    x=embedding.T[0]
    y=embedding.T[1]

    alpha=10/(np.sqrt(embedding.shape[0]))
    print(f"embedplot alpha: {alpha}, {(np.sqrt(embedding.shape[0]))}")

    ### scatter plot with marginal axes
    sns.set_style('white')

    embed_plot = sns.jointplot(x=x, y=y,
                hue=categories, palette=palette,
                legend='full',
                lw=0,
                joint_kws = dict(alpha=alpha),           #FUTURE: scale alpha with log(n_pixels)
                height=12, ratio=6
                )

    handles, __ = embed_plot.ax_joint.get_legend_handles_labels()

    embed_plot.ax_joint.legend(handles=handles, labels=labels, fontsize=10)

    embed_plot.set_axis_labels('x', 'y', fontsize=16)

    xmin=np.min(embedding[:,0])
    xmax=np.max(embedding[:,0])
    if xmax < 0:
        print("WARNING: xmax < 0, plot limits may show unexpected behaviour")

    embed_plot.ax_marg_x.set_xlim(xmin, xmax+(xmax-xmin)*0.15)

    sns.despine(ax=None, left=True, bottom=True)

    fig = embed_plot.fig

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

    #drop the floor slightly to emphasise low-but-nonzero regions
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


def contours_2d(kde):

    #lower the floor slightly to emphasise low-but-nonzero regions
    Z_local = np.copy(kde.Z)
    Z_local[Z_local < 0.00001] = -0.0001    #-0.00000001 if viridis

    fig = plt.figure(figsize=(24,18))
    ax = fig.add_subplot()

    cfset = ax.contourf(kde.X, kde.Y, Z_local, levels=25, cmap='Blues')

    ## ALT direct kernel density estimate plot
    #   flipped vertically
    #ax.imshow(Z_local, cmap='Blues', extent=[-7, 20, -7, 20])    

    #ADD contour lines
    #cset = ax.contour(kde.X, kde.Y, Z_local, colors='k')

    #ADD contour labels
    #ax.clabel(cset, inline=1, fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return fig



def plot_clusters(categories, classavg, embedding, kde, dims, output_directory=".", plot_kde=False, plot_margins=False, labels=[]):
    """
    display all plots for clusters
    """

    if not labels == []:
        df = tabular.get_df(classavg, labels)
        major_list = tabular.get_major_list(df)
        class_labels = tabular.nestlist_as_str(major_list)
    else:
        class_labels = []

    print(
    "---------------------------\n"
    "VISUALISATION\n"
    "---------------------------\n"
    )
    if embedding.shape[1] == 2:
        print("using 2d embedding x") 
        #generate the palette from the categories, independent of distance
        palette=colours.build_aligned_palette(embedding, categories)
        embedding_2d = embedding
    else:
        #use the 3D embedding to colour the categories based on distance
        fig_embed_map = embedding_map(embedding, dims)
        fig_embed_map.savefig(os.path.join(output_directory,'vis_embed_map.png'), transparent=False)  

        # produce 2D embedding for visualisation
        print("creating 2d embedding")
        ___, embedding_2d = clustering.reduce(embedding, "PCA", target_components=2) 
        palette=sns.color_palette(rgb_from_centroids(embedding, categories))

    if plot_margins:
        print("saving map with margins")        
        fig_cat_map = category_map(categories, dims, palette=palette)
        fig_cat_map.savefig(os.path.join(output_directory,'vis_category_map.png'), transparent=False)    
    else:
        print("creating category map")
        fig_cat_map = category_map_direct(categories, dims, palette=palette)
        fig_cat_map.savefig(os.path.join(output_directory,'vis_category_map.png'), transparent=False)  

    print("creating embedplot")    

    fig_embed = seaborn_embedplot(embedding_2d, categories, palette=palette, labels=class_labels)
    fig_embed.savefig(os.path.join(output_directory,'vis_embeddings.png'), transparent=False)    
    
    if plot_kde and kde is not None:
        fig_contours_3d = contours_3d(kde)
        fig_contours_2d = contours_2d(kde)
        fig_contours_2d.savefig(os.path.join(output_directory,'vis_kde.png'), transparent=False)
    
    tabular.printout(df)

    #plt.show()

    return palette


def plot_classes(categories, labels, classavg, palette):
    """
    display details for categories
    
    """
    category_avgs(categories, labels, classavg, palette=palette)    