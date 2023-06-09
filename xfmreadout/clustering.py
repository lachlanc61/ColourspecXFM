import time
import os
import re
import hdbscan
import numpy as np
import umap.umap_ as umap

from sklearn import decomposition
from sklearn.cluster import KMeans

#-----------------------------------
#CONSTANTS
#-----------------------------------
#REDUCERS
UMAP_COMPONENTS=2
PCA_COMPONENTS=2
UMAP_LOW_MEM=False
UMAP_VERBOSE=False

#DEFAULTS
DBSCAN_E=0.5   #epsilon: do not separate clusters closer than this value - refer umap min_dist
DBSCAN_CSIZE=200    #minimum cluster size
DBSCAN_MINSAMPLES=100   #minimum samples - larger = more conservative, more unclustered points
DBSCAN_METHOD="eom"

#BEST
DBSCAN_E=0.1   #epsilon: do not separate clusters closer than this value - refer umap min_dist
DBSCAN_CSIZE=1000  #minimum cluster size
DBSCAN_MINSAMPLES=500   #minimum samples - larger = more conservative, more unclustered points
DBSCAN_METHOD="leaf"

#KMEANS
KMEANS_CLUSTERS=10

#-----------------------------------
#GROUPS
#-----------------------------------
REDUCERS = [
    (decomposition.PCA, {"n_components": 2}),
    (umap.UMAP, {"n_components":2, 
        "n_neighbors": 300, 
        "min_dist": 0.1, 
        "low_memory": UMAP_LOW_MEM, 
        "verbose": True}),
]




CLUSTERERS = [
    (KMeans, {"init":"random", 
        "n_clusters": KMEANS_CLUSTERS, 
        "n_init": KMEANS_CLUSTERS, 
        "max_iter": 300, 
        "random_state": 42 }),

    (hdbscan.HDBSCAN, {"min_cluster_size": DBSCAN_CSIZE,
        "min_samples": DBSCAN_MINSAMPLES,
        "cluster_selection_epsilon": DBSCAN_E, 
        "cluster_selection_method": DBSCAN_METHOD,    
        "gen_min_span_tree": True }),
]


#-----------------------------------
#FUNCTIONS
#-----------------------------------

def get_operator_name(operator):
    """
    extract name of operator from the object
    eg. extracts "UMAP" from <class umap.umap_.UMAP>
    """
    if type(operator) == type:
        return repr(operator()).split("(")[0]
    else:
        return repr(operator).split("(")[0]


def find_operator(list, target_name: str):
    """
    search for a particular operator in a list
    """
    for operator, args in list:
        opname=get_operator_name(operator)
        if re.search(target_name,opname):
            return operator, args
    raise ValueError(f"{target_name} not a valid operator")


def multireduce(data, target_components=2):
    """
    perform dimensionality reduction
    args:       data
    returns:    embedding matrix, time per cluster
    """  
    UMAP_CUTOFF=50000000
    CHAN_CUTOFF=31

    reducer_list=REDUCERS
    npx=data.shape[0]
    nchan=data.shape[1]

    start_time = time.time()

    if nchan >= CHAN_CUTOFF:
        operator, args = find_operator(reducer_list, "PCA")
        args["n_components"]=target_components
        
        reducer = operator(**args)
        embedding = reducer.fit_transform(data)
    else:
        operator, args=find_operator(reducer_list, "UMAP")
        args["n_components"]=target_components        
        
        reducer = operator(**args)
        embedding = reducer.fit_transform(data)

    clusttimes = time.time() - start_time

    return reducer, embedding, clusttimes


def doclustering(embedding):
    """
    performs clustering on embedding to produce final clusters

    args:       set of 2D embedding matrices (shape [nreducers,x,y]), number of pixels in map
    returns:    category-by-pixel matrix, shape [nreducers,chan]
    """

    print("RUNNING CLASSIFIER")
    classifier_list = CLUSTERERS

    operator, args = find_operator(classifier_list, "HDBSCAN")
        
    classifier = operator(**args)
    embedding = classifier.fit(embedding)

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

def run(data, output_dir: str, force_embed=False, force_clust=False, overwrite=True, sqrt=True):

    if force_embed:
        force_clust = True

    #start a timer
    starttime = time.time() 

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

    if sqrt:
        data=np.sqrt(data)

    #   produce reduced-dim embedding per reducer
    if force_embed or not exists_embed:
        print("CALCULATING EMBED")
        reducer, embedding, clusttimes = multireduce(data)
        if overwrite or not exists_embed:
            np.save(file_embed,embedding)
        if overwrite or not exists_embed:
            np.save(file_ctime,clusttimes)
    else:
        print("LOADING EMBED")
        embedding = np.load(file_embed)
        #clusttimes = np.load(file_ctime)     
        clusttimes = None   


    #   calculate clusters from embedding
    if force_clust or not exists_cats:
        print("CALCULATING CATS")        
        classifier, categories = doclustering(embedding)
        if overwrite or not exists_cats:
            np.save(file_cats,categories)
    else:
        print("LOADING CATS")
        categories = np.load(file_cats)
        classifier = None

    #   sum and extract class averages
    n_clusters = count_categories(categories)
    classavg=np.zeros([len(REDUCERS),n_clusters, n_channels])

    if force_clust or not exists_classes:
        classavg=sumclusters(data, categories, n_clusters, n_channels) 
        if overwrite or not exists_classes:
            np.save(file_classes,classavg)
    else:
        classavg = np.load(file_classes)

    #complete the timer
    runtime = time.time() - starttime

    print(
    "---------------------------\n"
    "CLASSIFICATION COMPLETE\n"
    "---------------------------\n"
    f"total time: {round(runtime,2)} s\n"
    f"time per pixel: {round((runtime/totalpx),6)} s\n"
    "---------------------------"
    )

    return categories, classavg, embedding, clusttimes, classifier


#-----------------------------------
#INITIALISE
#-----------------------------------

