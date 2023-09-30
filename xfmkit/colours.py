
import copy
import numpy as np
import seaborn as sns
import colorcet as cc

import xfmkit.utils as utils
import xfmkit.clustering as clustering
import xfmkit.somfit as somfit

import logging
logger = logging.getLogger(__name__)



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
    else:
        cmapname=cc.glasbey_light
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

def build_aligned_palette(embedding, categories):
    """
    generates a palette for categories

    coloured by distance between cluster centroids
        by pulling from a colour embedding

    applies grey to unassigned/negative
    """

    cat_min=np.min(categories)
    cat_max=np.max(categories)
    n_cats=cat_max-cat_min+1
    n_colours = n_cats*3

    print("compile centroids")
    
    category_centroids = utils.compile_centroids(embedding, categories)

    print("embed colormap")

    new_palette, new_palette_embedding = embed_colourmap(n_colours=n_colours) 

    print("norm onto 2d")

    palette_embedding = utils.norm_onto_2d(new_palette_embedding, embedding)

    print("get closest")

    palette_indices = utils.get_closest_points(palette_embedding, category_centroids)

    print("palette from indices")

    final_palette = palette_from_indices(new_palette, palette_indices)
    
    return final_palette


def shuffle_palette(palette):
    """
    shuffles palette into N blocks of spaced sequential colours
        ie. breaks up a sequential colour sequence
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


def palette_from_indices(palette, indices):

    GREY=( 0.5, 0.5, 0.5 )

    result = copy.deepcopy(palette)

    del result[0:]

    for i in indices:        
        result.append(palette[i])

    del result[0]

    result.insert( 0, GREY )    

    return result


def embed_colourmap(n_colours=99):
    """
    create a colourmap embedding
    """

    print(n_colours)

    palette = sns.color_palette(cc.glasbey_light,n_colours)
    colours = np.array(palette, dtype=np.float32)

    # produce 2D embedding for visualisation
    ___, colour_embedding = clustering.reduce(colours, "UMAP", target_components=2) 
    
    colour_embedding__ = np.copy(colour_embedding)
    colour_embedding__ = colour_embedding__-np.min(colour_embedding__)    

    print(len(palette), len(colour_embedding))

    return palette, colour_embedding

def som_colourmap():
    cc_palette=sns.color_palette(cc.glasbey_light,100)

    linear_colours = somfit.som_on_palette(cc_palette)

    som_palette = copy.deepcopy(cc_palette)

    del som_palette[0:]

    for i in range(linear_colours.shape[0]):
        som_palette.append(linear_colours[i])

    return som_palette