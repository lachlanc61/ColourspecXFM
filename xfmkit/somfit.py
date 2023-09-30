import time
import os
import re
import copy
import pickle
import numpy as np
from minisom import MiniSom

import numpy as np

import xfmkit.config as config

m = config.get('som', 'default_neurons_m')
n = config.get('som', 'default_neurons_n')
default_steps = config.get('som', 'default_steps')


"""
minisom params:
sigma spread up update around winning node
basically spread of gaussian or cutoff for bubble
units are in x,y  (ie. 4x4 neurons, sig=1 neuron )

to use winner-takes-all (ie. only winning neuron updated)
neighbourhood = triangle, sigma=1

"""

def som_on_palette(cc_palette):

    coloursom = MiniSom(m, n, len(cc_palette[0]), sigma=1.0,
                learning_rate=0.2, neighborhood_function='gaussian')

    coloursom.train(cc_palette, int(round(default_steps/10)), random_order=True, verbose=True)

    _winners = coloursom.win_map(cc_palette, return_indices=False)

    neuron_colours=np.zeros((m,n,3), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            #print(f"({i},{j})")
            #print(len(_winners[(i,j)]))
            neuron_colours[i,j,:]=_winners[(i,j)][0]

    linear_colours = np.reshape(neuron_colours, (m*n,-1))

    return linear_colours


def categories_by_som(data):
# SOM initialization and training
    print('training...')
    som = MiniSom(m, n, data.shape[1], sigma=0.5,
                learning_rate=0.1, neighborhood_function='gaussian')
    som.random_weights_init(data)
    starting_weights = som.get_weights().copy()  # saving the starting weights

    som.train(data, default_steps, random_order=True, verbose=True)

    #flatten neuron indices
    som_categories=np.zeros(data.shape[0], dtype=np.int32)

    for i in range(data.shape[0]):
        _h, _k = som.winner(data[i])
        som_categories[i]=_h*m+_k

    """
    #FUTURE: better structure from minisom examples
    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)    
    """

    return som, som_categories




def run(data, output_dir: str, force=False, overwrite=True):

    #start a timer
    starttime = time.time() 

    file_embed=os.path.join(output_dir,f"embedding_som.pickle")
    file_cats=os.path.join(output_dir,"categories.npy")
    file_classes=os.path.join(output_dir,"classavg.npy")

    exists_embed = os.path.isfile(file_embed)
    exists_cats = os.path.isfile(file_cats)
    exists_classes = os.path.isfile(file_classes)

    totalpx = data.shape[0]
    n_channels = data.shape[1]

    #   produce reduced-dim embedding per reducer
    if force or not ( exists_embed and exists_cats):
        print("FITTING SOM")
        som, categories = categories_by_som(data)

        if overwrite or not exists_embed:
            print("Pickling SOM") 
            pickle.dump(som, open(file_embed, "wb"))   


        if overwrite or not exists_cats:
            np.save(file_cats,categories)        
        print("COMPLETED SOM")
    else:
        print("LOADING SOM")
        categories = np.load(file_cats)
        som = pickle.load(open(file_embed, "rb"))

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

    return categories, som, None