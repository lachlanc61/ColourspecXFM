import copy
import logging
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from matplotlib import colors
from PIL import Image



REDUCER=1

logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)


def remap(indata, dims):
    """
    restores map from linear data + map dimensions
    """
    print(indata.shape)
    if np.shape(indata.shape)[0] == 2:
        return indata.reshape(dims[0], dims[1], -1)
    else:
        return indata.reshape(dims[0], -1)



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


def show_map(data, elements, dims, idx):

    maps=remap(data, dims)

    img = maps[:,:,idx]

    print(elements[idx])
    print(np.quantile(img, 0.1), np.quantile(img, 0.5), np.quantile(img, 0.9))

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(111)

    display = ax.imshow(img, cmap='plasma')

    plt.show()

    return

def category_map ( categories, data, dims, palette=None ):
    """
    image of categories
    """

    #KCMAPS=["tab10"]    #colourmaps for kmeans

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    ncats=np.max(categories)+2
    print(ncats)

    #axcm=cm.get_cmap(KCMAPS[0], ncats)

    #cmap=axcm(range(ncats))
    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=build_palette(categories)

    cmap = colors.ListedColormap(palette)
    #reshape the category list back to the map dimensions using xdim
    #WARNING: fails using SHORTRUN unless ends at end of row - fix this later

    catmap=remap(categories+1,dims)

    print(np.min(categories))
    print(np.min(catmap))

    #show this category image
    ax.imshow(catmap, cmap=cmap)

    return

    #fig.savefig(os.path.join(EMBED_DIR,"cluster_map.png"), dpi=200)

def category_avgs(categories, elements, classavg, palette=None ):
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
    category:element boxplots
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

def seaborn_embedplot(embedding, categories, palette=None):


    if palette is None:
        log.warning(f"palette not given, building from categories")
        palette=build_palette(categories)

    x=embedding.T[0]
    y=embedding.T[1]

    ### scatter plot with marginal axes
    sns.set_style('white')

    sns.jointplot(x=x, y=y,
                hue=categories, palette=palette,
                lw=0,
                joint_kws = dict(alpha=0.01),
                height=10, ratio=6
                )

    #xlim=[-3,3], ylim=[-3,3],

    ax = sns.despine(ax=None, left=True, bottom=True)
    #plt.savefig('j_scatter_tr.png', transparent=True)
    plt.show()

    return

def seaborn_kdeplot(embedding, categories):
    """
    kde plot filled with colors with transparent background

    """
    x=embedding.T[0]
    y=embedding.T[1]
    """
    sns.set_style('white')
    ax = sns.jointplot(x=x, y=y,
                cut = 0, hue=categories,
                palette=sns.color_palette("dark"),
                kind='kde', fill=True,
                height=15, ratio=6,
                joint_kws = dict(alpha=0.4),
                marginal_kws=dict(fill=True),
                legend=False)
    ax.ax_marg_x.remove()
    ax.ax_marg_y.remove()
    ax = sns.despine(ax=None, left=True, bottom=True)
    #plt.savefig('kde_tr_fill.png', transparent=True)
    """
    sns.set_style('white')
    ax = sns.kdeplot(x=x, y=y,
                hue=categories,
                fill=True,
                legend=False)
#    ax.ax_marg_x.remove()
#    ax.ax_marg_y.remove()
    
    #ax = sns.despine(ax=None, left=True, bottom=True)



    plt.show()

#

def seaborn_kdecontours(embedding, categories):
    """
    kde plot filled with colors with transparent background

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