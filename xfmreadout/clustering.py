import numpy as np
import matplotlib.pyplot as plt
import time
import os

from matplotlib import cm
from sklearn import decomposition
from sklearn.cluster import KMeans
import umap.umap_ as umap

import xfmreadout.utils as utils

#-----------------------------------
#CONSTANTS
#-----------------------------------
#KCMAPS=["Accent","Set1"]    #colourmaps for kmeans
KCMAPS=["tab10","tab10"]    #colourmaps for kmeans

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
reducers = [
    (decomposition.PCA, {}),
#    (decomposition.IncrementalPCA, {"batch_size": 10000}),
#    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3, "low_memory": True, "verbose": True}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3, "low_memory": True}),
]


#-----------------------------------
#FUNCTIONS
#-----------------------------------

def getredname(i):
    """
    get name of reducer from specified index
    args:       index of reducer
    returns:    reducer name
    """
    return repr(reducers[i][0]()).split("(")[0]

def reduce(data):
    """
    performs dimensionality reduction on data using reducers
    args:       data
    returns:    embedding matrix, time per cluster
    """
    npx=data.shape[0]
    embedding=np.zeros((nred,npx,2))
    clusttimes=np.zeros(nred)

    i = 0
    for reducer, args in reducers:
        redname=getredname(i)
        start_time = time.time()

        print(f'REDUCER {i+1} of {nred}: {redname} across {npx} elements')

        #do it
        embed = reducer(n_components=2, **args).fit_transform(data)
        
        clusttimes[i] = time.time() - start_time
        embedding[i,:,:]=embed
        i += 1
    return embedding, clusttimes


def dokmeans(embedding, npx, n_clusters):
    """
    performs kmeans on embedding matrices to cluster 2D matrices from reducers 

    args:       set of 2D embedding matrices (shape [nreducers,x,y]), number of pixels in map
    returns:    category-by-pixel matrix, shape [nreducers,chan]
    """
    #initialise kwargs
    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=n_clusters,
        max_iter=300,
        random_state=42
    )

    categories=np.zeros((nred,npx),dtype=np.uint16)
    for i in np.arange(0,nred):
        redname=repr(reducers[i][0]()).split("(")[0]
        embed = embedding[i,:,:]

        print(f'KMEANS clustering {i+1} of {nred}, reducer {redname} across {npx} elements')

        #DO:
        kmeans.fit(embed)
        categories[i]=kmeans.labels_

    return categories

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
    fig, (ax) = plt.subplots(nred, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.tight_layout(pad=2)
 
    #fig.subplots_adjust(
    #    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    #)

    labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}

    #for each reducer
    for i in np.arange(0,nred):
        #get the reducer's name
        redname=repr(reducers[i][0]()).split("(")[0]
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

def calculate(data, totalpx, n_clusters, n_channels):

    #   produce reduced-dim embedding per reducer
    embedding, clusttimes = reduce(data)

    #   cluster via kmeans on embedding
    categories = dokmeans(embedding, totalpx, n_clusters)

    #produce and save cluster averages

    #   initialise averages
    classavg=np.zeros([len(reducers),n_clusters, n_channels])

    #   cycle through reducers
    for i in range(len(reducers)):
        classavg[i]=sumclusters(data, categories[i], n_clusters, n_channels)    

    return categories, classavg, embedding, clusttimes


def complete(categories, classavg, embedding, clusttimes, energy, mapx, mapy, n_clusters, dirs ):

    #   cycle through reducers
    for i in range(len(reducers)):
        redname=getredname(i)

        #saving embeddings
        np.savetxt(os.path.join(dirs.transforms, redname + ".dat"), embedding[i,:,:])

        #saving kmeans categories
        np.savetxt(os.path.join(dirs.transforms, redname + "_kmeans.txt"), categories[i])

        #saving individual cluster averages
        for j in range(n_clusters):
            print(f'saving reducer {redname} cluster {j} with shape {classavg[i,j,:].shape}', end='\r')
            np.savetxt(os.path.join(dirs.transforms, "sum_" + redname + "_" + str(j) + ".txt"), np.c_[energy, classavg[i,j,:]], fmt=['%1.3e','%1.6e'])
       
        print(f'saving combined file for {redname}')
        np.savetxt(os.path.join(dirs.transforms, "sum_" + redname + ".txt"), np.c_[energy, classavg[i,:,:].transpose(1,0)], fmt='%1.5e')             
        #plt.plot(energy, clustaverages[i,j,:])
    
    fig = clustplt(embedding, categories, mapx, clusttimes)

    #save and show
    fig.savefig(os.path.join(dirs.plots, 'clusters.png'), dpi=150)

    return 

#-----------------------------------
#INITIALISE
#-----------------------------------

nred = len(reducers)

