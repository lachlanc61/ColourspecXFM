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
    perform dimensionality reduction
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
    #DEFAULTS
    DBSCAN_E=0.1   #smaller clusters
    DBSCAN_CSIZE=200
    DBSCAN_MINSAMPLES=100


    #DBSCAN_E=0.5    #really large clusters
    #DBSCAN_E=0.1   #many small clusters
    DBSCAN_E=0.001   #smaller clusters
    DBSCAN_CSIZE=200
    DBSCAN_MINSAMPLES=100

    #initialise clustering options
    kmeans = KMeans(
        init="random",
        n_clusters=N_CLUSTERS,
        n_init=N_CLUSTERS,
        max_iter=300,
        random_state=42
    )
    dbscan = hdbscan.HDBSCAN(
        min_cluster_size=DBSCAN_CSIZE,
        min_samples=DBSCAN_MINSAMPLES,
        cluster_selection_epsilon=DBSCAN_E,
        gen_min_span_tree=True
    )


    print("RUNNING CLUSTERING")
    print(f"DBSCAN PARAM {DBSCAN_E}, {dbscan.cluster_selection_epsilon}")    
    classifier = dbscan

    categories=np.zeros((npx),dtype=np.uint16)

    classifier.fit(embedding)

    categories=classifier.labels_

    return classifier, categories

def sumclusters(dataset, categories, n_clusters, n_channels):
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

    if n_clusters != count_categories(categories):
        raise ValueError("cluster count mismatch")

    for i in range(np.min(categories), np.max(categories)):
        datcat=dataset[categories==i]
        print(f"cluster {i}, count: {datcat.shape[0]}") #DEBUG
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
    pass

def count_categories(categories):
    """
    return the total number of categories, including negative values
    """
    min_cat = np.min(categories)
    max_cat = np.max(categories)
    num_cats = max_cat - min_cat + 1

    return num_cats


def complete(categories, classavg, embedding, clusttimes, energy, mapx, mapy, n_clusters, dirs ):
       
    fig = clustplt(embedding, categories, mapx, clusttimes)

    #save and show
    fig.savefig(os.path.join(dirs.plots, 'clusters.png'), dpi=150)

    return 

def get(data, output_dir: str, force_embed=False, force_clust=False, overwrite=False):
    file_embed=os.path.join(output_dir,"embedding.npy")
    file_cats=os.path.join(output_dir,"categories.npy")
    file_classes=os.path.join(output_dir,"classavg.npy")
    file_ctime=os.path.join(output_dir,"clusttimes.npy")

    exists_embed = os.path.isfile(file_embed)
    exists_cats = os.path.isfile(file_cats)
    exists_classes = os.path.isfile(file_classes)
    exists_ctime = os.path.isfile(file_ctime)

    totalpx = data.shape[0]
    n_channels = data.shape[1]

    #   produce reduced-dim embedding per reducer
    if force_embed or not exists_embed:
        print("CALCULATING EMBED")
        reducer, embedding, clusttimes = reduce(data)
        if overwrite or not exists_embed:
            np.save(file_embed,embedding)
        if overwrite or not exists_embed:
            np.save(file_ctime,clusttimes)
    else:
        print("LOADING EMBED")
        embedding = np.load(file_embed)
        clusttimes = np.load(file_ctime)        


    #   calculate clusters from embedding
    if force_clust or not exists_cats:
        print("CALCULATING CATS")        
        classifier, categories = doclustering(embedding, totalpx)
        if overwrite or not exists_cats:
            np.save(file_cats,categories)
    else:
        print("LOADING CATS")
        categories = np.load(file_cats)

    #   sum and extract class averages
    n_clusters = count_categories(categories)
    classavg=np.zeros([len(REDUCERS),n_clusters, n_channels])

    if force_clust or not exists_classes:
        classavg=sumclusters(data, categories, n_clusters, n_channels) 
        if overwrite or not exists_classes:
            np.save(file_classes,classavg)
    else:
        classavg = np.load(file_classes)

    return categories, classavg, embedding, clusttimes


#-----------------------------------
#INITIALISE
#-----------------------------------

