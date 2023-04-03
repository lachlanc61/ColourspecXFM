import numpy as np
import matplotlib.pyplot as plt
import time
import os

from matplotlib import cm
from sklearn import decomposition

from sklearn.cluster import KMeans

import umap.umap_ as umap
import hdbscan

import xfmreadout.utils as utils

#-----------------------------------
#CONSTANTS
#-----------------------------------
#KCMAPS=["Accent","Set1"]    #colourmaps for kmeans
KCMAPS=["tab10","tab10"]    #colourmaps for kmeans
N_CLUSTERS=10

#-----------------------------------
#LISTS
#-----------------------------------
"""
#full reducer list here
from sklearn import datasets, decomposition, manifold, preprocessing

reducers = [
    (manifold.TSNE, {"perplexity": 50}),
    # (manifold.LocallyLinearEmbedding, {'n_neighbors':10, 'method':'hessian'}),
    (manifold.Isomap, {"n_neighbors": 30}),
    (manifold.MDS, {}),
    (decomposition.PCA, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
]
"""
REDUCERS = [
    (decomposition.PCA, {}),
#    (decomposition.IncrementalPCA, {"batch_size": 10000}),
#    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3, "low_memory": True, "verbose": True}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3, "low_memory": True, "verbose": True}),
]


CLUSTERERS = [
    (KMeans, {"init":"random", "n_clusters": N_CLUSTERS, "n_init": N_CLUSTERS, \
              "max_iter": 300, "random_state": 42 }),

    (hdbscan.HDBSCAN, {"min_cluster_size": 200,
        "min_samples": 100,
        "cluster_selection_epsilon": 0.01,
        "gen_min_span_tree": True }),
]



#-----------------------------------
#FUNCTIONS
#-----------------------------------

def getobjname(obj):
    """
    get name of reducer from specified index
    args:       index of reducer
    returns:    reducer name
    """
    if type(obj) == type:
        return repr(obj()).split("(")[0]
    else:
        return repr(obj).split("(")[0]

def reduce(data):
    """
    performs dimensionality reduction on data using reducers
    args:       data
    returns:    embedding matrix, time per cluster
    """

    n_components = 2

    #initialise reducer options
    pca= decomposition.PCA(n_components=n_components)

    umapper = umap.UMAP(
        n_components=n_components,
        n_neighbors=30, 
        min_dist=0.1, 
        low_memory=True, 
        verbose=True
    )

    reducer = umapper

    npx=data.shape[0]
    embedding=np.zeros((npx,n_components))

    redname=getobjname(reducer)
    start_time = time.time()

    print(f'Dimensionality reduction via {redname} across {npx} elements')

    #do it
    reducer.fit(data)
    
    embedding = reducer.transform(data)

    #_transform(data)
    
    clusttimes = time.time() - start_time

    return reducer, embedding, clusttimes


def doclustering(embedding, npx):
    """
    performs clustering on embedding to produce final clusters

    args:       set of 2D embedding matrices (shape [nreducers,x,y]), number of pixels in map
    returns:    category-by-pixel matrix, shape [nreducers,chan]
    """

    #DBSCAN_E=0.1    #many small clusters
    DBSCAN_E=0.01   #larger clusters


    #initialise clustering options
    kmeans = KMeans(
        init="random",
        n_clusters=N_CLUSTERS,
        n_init=N_CLUSTERS,
        max_iter=300,
        random_state=42
    )

    dbscan = hdbscan.HDBSCAN(
        min_cluster_size=200,
        min_samples=100,
        cluster_selection_epsilon=DBSCAN_E,
        gen_min_span_tree=True
    )

    classifier = dbscan

    categories=np.zeros((npx),dtype=np.uint16)

    classifier.fit(embedding)

    categories=classifier.labels_

    return classifier, categories

def sumclusters(dataset, catlist, n_clusters, n_channels):
    """
    calculate summed spectrum for each cluster
    args: 
        dataset, spectrum by px
        catlist, categories by px
    returns:
        specsum, spectrum by category
    
    aware: nclust, number of clusters
    """
    specsum=np.zeros((n_clusters,n_channels))

    for i in range(n_clusters):
        datcat=dataset[catlist==i]
        pxincat = datcat.shape[0]   #no. pixels in category i
        specsum[i,:]=(np.sum(datcat,axis=0))/pxincat
    return specsum

def clustplt(embedding, categories, mapx, clusttimes):
    """
    receives arrays from reducers and kmeans
    + time to cluster

    plots Nx2 plot for each reducer

    https://towardsdatascience.com/clearing-the-confusion-once-and-for-all-fig-ax-plt-subplots-b122bb7783ca
    """    
        
    #create figure and ax matrix
    #   gridspec adjusts widths of subplots in each row
    fig, (ax) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.tight_layout(pad=2)
 
    #fig.subplots_adjust(
    #    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    #)

    labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}

    #for each reducer
    for i in np.arange(0,1):
        #get the reducer's name
        redname=repr(REDUCERS[i][0]()).split("(")[0]
        #read in the embedding xy array and time
        embed = embedding[i,:]
        elapsed_time = clusttimes[i]
        
        #assign index in plot matrix
        plotid=(i,0)

        #adjust plotting options
        ax[plotid].set_xlabel(redname, size=16)
        ax[plotid].xaxis.set_label_position("top")

        #create the scatterplot for this reducer
        # .T = transpose, rotates x and y
        ax[plotid].scatter(*embed.T, s=10, c=categories[i], cmap=KCMAPS[i], alpha=0.25)

        #add the runtime as text
        ax[plotid].text(
            0.99,
            0.01,
            "{:.2f} s".format(elapsed_time),
            transform=ax[plotid].transAxes,
            size=14,
            horizontalalignment="right",
        )

        ncats=np.max(categories)+1
        axcm=cm.get_cmap(KCMAPS[i], ncats)

        cmap=axcm(range(ncats))

        for j in range(ncats):
            ax[plotid].text(
                0.2+0.067*j,
                -0.1,
                f"{j}",
                transform=ax[plotid].transAxes,
                size=14,
                horizontalalignment="right",
                color=cmap[j]
            )
        
        #assign index for category map for this reducer
        plotid=(i,1)

        #reshape the category list back to the map dimensions using xdim

        catmap=np.reshape(categories[i], [-1, mapx])
        
        #show this category image
        ax[plotid].imshow(catmap, cmap=KCMAPS[i])


    #initalise the final plot, clear the axes
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()
 
    return fig

def calculate(data):

    totalpx = data.shape[0]
    n_channels = data.shape[1]

    #   produce reduced-dim embedding per reducer
    reducer, embedding, clusttimes = reduce(data)

    #   cluster via kmeans on embedding
    classifier, categories = doclustering(embedding, totalpx)

    #produce and save cluster averages

    n_clusters = np.max(categories)+1

    #   initialise averages
    classavg=np.zeros([len(REDUCERS),n_clusters, n_channels])

    classavg=sumclusters(data, categories, n_clusters, n_channels)    

    return categories, classavg, embedding, clusttimes


def complete(categories, classavg, embedding, clusttimes, energy, mapx, mapy, n_clusters, dirs ):
       
    fig = clustplt(embedding, categories, mapx, clusttimes)

    #save and show
    fig.savefig(os.path.join(dirs.plots, 'clusters.png'), dpi=150)

    return 

def get(data, output_dir: str, force=False, overwrite=True):

    file_cats=os.path.join(output_dir,"categories.npy")
    file_classes=os.path.join(output_dir,"classavg.npy")
    file_embed=os.path.join(output_dir,"embedding.npy")
    file_ctime=os.path.join(output_dir,"clusttimes.npy")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filesexist = os.path.isfile(file_cats) and os.path.isfile(file_classes) \
            and  os.path.isfile(file_embed) and os.path.isfile(file_ctime)
    
    if force or not filesexist:
        categories, classavg, embedding, clusttimes = calculate(data)
        #embedding, clusttimes = clustering.reduce(data)

        if overwrite:
            np.save(file_cats,categories)
            np.save(file_classes,classavg)
            np.save(file_embed,embedding)
            np.save(file_ctime,clusttimes)
    else:
        categories = np.load(file_cats)
        classavg = np.load(file_classes)
        embedding = np.load(file_embed)
        clusttimes = np.load(file_ctime)

    return categories, classavg, embedding, clusttimes

#-----------------------------------
#INITIALISE
#-----------------------------------

