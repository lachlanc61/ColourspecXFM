import time
import os
import re
import hdbscan
import numpy as np
import umap.umap_ as umap
import pacmap
import pickle

from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

import xfmkit.utils as utils
import xfmkit.config as config

import logging
logger = logging.getLogger(__name__)

#-----------------------------------
#CONSTANTS
#-----------------------------------
#REDUCERS
final_components=config.get('reducer', 'final_components')
umap_precomponents=config.get('reducer', 'umap_precomponents')
min_separation=config.get('reducer', 'min_separation')
default_kde_points=config.get('reducer', 'default_kde_points')

#   odd number of points apparently speeds up rendering via mpl.plot_surface

#-----------------------------------
#GROUPS
#-----------------------------------
REDUCERS = [
    (decomposition.PCA, {"n_components": 2}),

    (umap.UMAP, {"n_components":2, 
        "n_neighbors": 30,  #300 
        "min_dist": min_separation, 
        "low_memory": True, 
        "verbose": True}),

    (pacmap.PaCMAP, {"n_components":2,
        "n_neighbors": None,
        "verbose": True }),
]

CLASSIFIERS = [
    (KMeans, {"init":"random", 
        "n_clusters": 10, 
        "n_init": 10, 
        "max_iter": 300, 
        "random_state": 42 }),

    (hdbscan.HDBSCAN, {"min_cluster_size": 100,
        "min_samples": 500,
        "alpha": 1.0,
        "cluster_selection_epsilon": 0.2,
        "cluster_selection_method": "eom", #eom
        "gen_min_span_tree": True }),
]

"""
    (hdbscan.HDBSCAN, {"min_cluster_size": 1000,    #6000
        "min_samples": 500,
        "alpha": 1.0,
        "cluster_selection_epsilon": min_separation, 
        "cluster_selection_method": "leaf",     #alt: "eom"
        "gen_min_span_tree": True }),
"""


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



def reduce(data, reducer_name: str, target_components=final_components):
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


def multireduce(data, target_components=final_components):
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
        __reducer, __embedding = reduce(data, "PCA", umap_precomponents)   
        reducer, embedding = reduce(__embedding, "UMAP", target_components)        

    else:
        if True:
            #go ahead with UMAP
            reducer, embedding = reduce(data, "UMAP", target_components)
        if False:
            #go ahead with PaCMAP
            reducer, embedding = reduce(data, "PaCMAP", target_components)

    return reducer, embedding


def classify(embedding, constrain: bool = False, majors_only: bool = False):
    """
    performs classification on embedding to produce final clusters

    args:       set of 2D embedding matrices (shape [nreducers,x,y]), number of pixels in map
    returns:    category-by-pixel matrix, shape [nreducers,chan]
    """
    USE="HDBSCAN"

    print("RUNNING CLASSIFIER")
    classifier_list = CLASSIFIERS

    if majors_only:
        cluster_sizefactor=50
    else:
        cluster_sizefactor=300

    if USE=="HDBSCAN":
        operator, args = find_operator(classifier_list, USE)
        if not constrain:
            if majors_only:
                args["cluster_selection_epsilon"]=0.3
            else:
                args["cluster_selection_epsilon"]=0.2
            args["cluster_selection_method"]="eom"               
            print(f"cluster_selection_epsilon: {args['cluster_selection_epsilon']}")
        else:
            if majors_only:
                cluster_factor=50
            else:
                cluster_factor=300            
            args["min_cluster_size"]=round(embedding.shape[0]/cluster_sizefactor)
            args["cluster_selection_method"]="leaf"   
            print(f"min cluster size: {embedding.shape[0]/cluster_sizefactor}")

    elif USE=="DBSCAN":
        operator, args = find_operator(classifier_list, USE) 

    classifier = operator(**args)
    embedding = classifier.fit(embedding)

    categories=classifier.labels_

    categories = categories.astype(np.int32)

    return classifier, categories

def calc_classavg(data, categories):
    """
    calculate summed spectrum for each cluster
    args: 
        dataset, spectrum by px
        catlist, categories by px
    returns:
        specsum, spectrum by category
    """
    n_channels = data.shape[1]
    n_clusters, ___ = utils.count_categories(categories)

    result=np.zeros((n_clusters,n_channels))

    for i in range(0, n_clusters):
        data_subset=data[categories==i]
        pxincat = data_subset.shape[0]  #no. pixels in category i
        print(f"cluster {i}, count: {pxincat}") #DEBUG

        if pxincat > 0:
            result[i,:]=(np.mean(data_subset,axis=0))
        else:   #assign nan to any category with zero, avoids warning from np.mean
            result[i,:]=float("nan")
        
    return result


class KdeMap():
    def __init__(self, embedding, n=default_kde_points):
        self.kde = KernelDensity(kernel='gaussian',bandwidth=min_separation*3)
        self.n = n

        print("Fitting KDE")
        self.kde.fit(embedding)

        print("Creating KDE")
        xy_, self.X, self.Y = get_linspace(embedding, self.n)        
        self.dimensions = self.X.shape
        self.Z = self.kde.score_samples(xy_)

        self.Z = np.exp(self.Z)
        self.Z = self.Z.reshape(self.X.shape)    
        print("KDE complete")


def get_linspace(embedding, n=default_kde_points):
    ex = embedding[:,0]
    ey = embedding[:,1]

    x = np.linspace(np.min(ex)-round(np.max(ex)/10), np.max(ex)+round(np.max(ex)/10), n)
    y = np.linspace(np.min(ey)-round(np.max(ey)/10), np.max(ey)+round(np.max(ey)/10), n)

    X, Y = np.meshgrid(x, y)

    xy = np.vstack([X.ravel(), Y.ravel(),]).T

    return xy, X, Y  

def get_classavg(raw_data, categories, output_dir, overwrite=True, labels=[]):

    file_classes=os.path.join(output_dir,"classavg.npy")
    csv_classes=os.path.join(output_dir,"class_averages.csv")
    exists_classes = os.path.isfile(file_classes)

    totalpx = raw_data.shape[0]
    n_channels = raw_data.shape[1]

    #   sum and extract class averages
    n_clusters, ___ = utils.count_categories(categories)

    classavg=calc_classavg(raw_data, categories)

    print("WRITING CLASS AVERAGES")
    if overwrite or not exists_classes:

        header=''
        if not labels == []:
            for i in range(raw_data.shape[1]):
                header=header+f"{labels[i]},"

        np.save(file_classes,classavg)
        np.savetxt(csv_classes, classavg, header=header, fmt='%.8f', delimiter=',')
    
    return classavg


def run(data, output_dir: str, constrain=False, majors=False, force_embed=False, force_clust=False, overwrite=True, target_components=2, do_kde=False):

    if force_embed:
        force_clust = True

    #start a timer
    starttime = time.time() 

    file_embed=os.path.join(output_dir,f"embedding_{target_components}d.npy")
    file_cats=os.path.join(output_dir,"categories.npy")
    file_classes=os.path.join(output_dir,"classavg.npy")
    file_kde=os.path.join(output_dir,f"kde_{target_components}d.pickle")

    exists_embed = os.path.isfile(file_embed)
    exists_cats = os.path.isfile(file_cats)
    exists_classes = os.path.isfile(file_classes)
    exists_kde = os.path.isfile(file_kde)

    totalpx = data.shape[0]
    n_channels = data.shape[1]

    #   produce reduced-dim embedding per reducer
    if force_embed or not exists_embed:
        print("CALCULATING EMBEDDING")
        reducer, embedding = multireduce(data, target_components=target_components)
        force_clust = True
        if overwrite or not exists_embed:
            np.save(file_embed,embedding)
        print("COMPLETED EMBEDDING")
    else:
        print("LOADING EMBEDDING")
        embedding = np.load(file_embed)
        #clusttimes = np.load(file_ctime)     

    #   calculate kde from embedding
    if do_kde and target_components == 2:
        if force_embed or not exists_kde:
            print(f"CALCULATING KDE with n={default_kde_points}")        
            kde = KdeMap(embedding, n=default_kde_points)
            if overwrite or not exists_kde:
                print("Pickling KDE") 
                pickle.dump(kde, open(file_kde, "wb"))
            print("COMPLETED KDE")
        else:
            print("LOADING KDE")
            kde = pickle.load(open(file_kde, "rb"))
    else:
        kde = None

    #   calculate clusters from embedding
    if force_clust or not exists_cats:
        print("CALCULATING CLASSIFICATION")        
        classifier, categories = classify(embedding, constrain=constrain, majors_only=majors)
        categories=categories+1     
        print(f"number of categories: {np.max(categories)}")
        if overwrite or not exists_cats:
            np.save(file_cats,categories)
    else:
        print("LOADING CLASSIFICATION")
        categories = np.load(file_cats)

        #if old category format with negative classes, update and re-save
        if np.min(categories) == -1:
            categories = categories+1
            np.save(file_cats,categories)

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

    return categories, embedding, kde


#-----------------------------------
#INITIALISE
#-----------------------------------

