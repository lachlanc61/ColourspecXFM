import time
import os
import re
import hdbscan
import numpy as np
import umap.umap_ as umap

from sklearn import decomposition
from sklearn.cluster import KMeans

import xfmreadout.utils as utils

#-----------------------------------
#CONSTANTS
#-----------------------------------
#REDUCERS
FINAL_COMPONENTS=2
UMAP_PRECOMPONENTS=11

UMAP_LOW_MEM=True
UMAP_VERBOSE=True


#CLASSIFIERS:
EST_N_CLUSTERS=30

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
        "n_neighbors": 30,  #300 
        "min_dist": 0.1, 
        "low_memory": UMAP_LOW_MEM, 
        "verbose": True}),
]

CLASSIFIERS = [
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



def reduce(data, reducer_name: str, target_components=FINAL_COMPONENTS):
    """
    perform dimensionality reduction using a specific reducer
    args:       data, reducer_name ("PCA", "UMAP"), target components
    returns:    reducer and embedding matrix
    """  
    reducer_list=REDUCERS

    operator, args = find_operator(reducer_list, reducer_name)
    args["n_components"]=target_components
    print(f"running reducer: {reducer_name} across data with shape: {data.shape}")

    reducer = operator(**args)
    embedding = reducer.fit_transform(data)    

    return reducer, embedding


def multireduce(data, target_components=FINAL_COMPONENTS):
    """
    manage dimensionality reduction based on size of dataset
    """  
    PXNR_CUTOFF=5000000
    DIMENSIONALITY_CUTOFF=31

    npx=data.shape[0]
    nchan=data.shape[1]

    start_time = time.time()

    if npx >= PXNR_CUTOFF:
        #if number of pixels is very high, use PCA
        reducer, embedding = reduce(data, "PCA", target_components)   

    elif nchan >= DIMENSIONALITY_CUTOFF:
        #if dimensionality is high, chain PCA into UMAP
        __reducer, __embedding = reduce(data, "PCA", UMAP_PRECOMPONENTS)   
        reducer, embedding = reduce(__embedding, "UMAP", target_components)        

    else:
        #go ahead with UMAP
        reducer, embedding = reduce(data, "UMAP", target_components)

    return reducer, embedding


def classify(embedding):
    """
    performs classification on embedding to produce final clusters

    args:       set of 2D embedding matrices (shape [nreducers,x,y]), number of pixels in map
    returns:    category-by-pixel matrix, shape [nreducers,chan]
    """

    print("RUNNING CLASSIFIER")
    classifier_list = CLASSIFIERS

    operator, args = find_operator(classifier_list, "HDBSCAN")

    args["min_cluster_size"]=round(embedding.shape[0]/EST_N_CLUSTERS)

    classifier = operator(**args)
    embedding = classifier.fit(embedding)

    categories=classifier.labels_

    return classifier, categories

def calc_classavg(data, categories, category_list, n_channels):
    """
    calculate summed spectrum for each cluster
    args: 
        dataset, spectrum by px
        catlist, categories by px
    returns:
        specsum, spectrum by category
    
    aware: nclust, number of clusters
    """
    n_channels = data.shape[1]
    n_clusters = len(category_list)

    result=np.zeros((n_clusters,n_channels))

    if n_clusters != utils.count_categories(categories)[0]:
        raise ValueError("cluster count mismatch")

    for i in range(n_clusters):
        icat=category_list[i]
        data_subset=data[categories==icat]
        pxincat = data_subset.shape[0]  #no. pixels in category i
        print(f"cluster {i}, count: {pxincat}") #DEBUG
        result[icat,:]=(np.sum(data_subset,axis=0))/pxincat
    return result

def clustplt(embedding, categories, mapx, clusttimes):
    pass




def complete(categories, classavg, embedding, clusttimes, energy, mapx, mapy, n_clusters, dirs ):
       
    fig = clustplt(embedding, categories, mapx, clusttimes)

    #save and show
    fig.savefig(os.path.join(dirs.plots, 'clusters.png'), dpi=150)

    return 


def get_classavg(raw_data, categories, output_dir, force=False, overwrite=True):

    file_classes=os.path.join(output_dir,"classavg.npy")
    exists_classes = os.path.isfile(file_classes)

    totalpx = raw_data.shape[0]
    n_channels = raw_data.shape[1]

    #   sum and extract class averages
    n_clusters, category_list = utils.count_categories(categories)
    classavg=np.zeros([len(REDUCERS),n_clusters, n_channels])

    if force or not exists_classes:
        classavg=calc_classavg(raw_data, categories, category_list, n_channels) 
        if overwrite or not exists_classes:
            np.save(file_classes,classavg)
    else:
        classavg = np.load(file_classes)
    
    return classavg


def run(data, output_dir: str, force_embed=False, force_clust=False, overwrite=True):

    if force_embed:
        force_clust = True

    #start a timer
    starttime = time.time() 

    file_embed=os.path.join(output_dir,"embedding.npy")
    file_cats=os.path.join(output_dir,"categories.npy")
    file_classes=os.path.join(output_dir,"classavg.npy")

    exists_embed = os.path.isfile(file_embed)
    exists_cats = os.path.isfile(file_cats)
    exists_classes = os.path.isfile(file_classes)

    totalpx = data.shape[0]
    n_channels = data.shape[1]

    #   produce reduced-dim embedding per reducer
    if force_embed or not exists_embed:
        print("CALCULATING EMBED")
        reducer, embedding = multireduce(data, target_components=3)
        if overwrite or not exists_embed:
            np.save(file_embed,embedding)
    else:
        print("LOADING EMBED")
        embedding = np.load(file_embed)
        #clusttimes = np.load(file_ctime)     

    #   calculate clusters from embedding
    if force_clust or not exists_cats:
        print("CALCULATING CATS")        
        classifier, categories = classify(embedding)
        if overwrite or not exists_cats:
            np.save(file_cats,categories)
    else:
        print("LOADING CATS")
        categories = np.load(file_cats)
        classifier = None

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

    return categories, embedding


#-----------------------------------
#INITIALISE
#-----------------------------------

