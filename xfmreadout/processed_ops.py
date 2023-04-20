import os
import re
import numpy as np
import periodictable as pt
import pandas as pd
from PIL import Image

import xfmreadout.clustering as clustering
import xfmreadout.processed_plots as processed_plots



FORCE = True
AUTOSAVE = True

EMBED_DIRNAME = "embedding"

IGNORE_ELEMENTS=['sum','Back','Compton','Mo','MoL']

TRUE_ELEMENTS = []

for ptelement in pt.elements:
    TRUE_ELEMENTS.append(ptelement.symbol)


def get_elements(files):
    """

    Extract element names and corresponding files

    Discard files that do not correspond to elements

    """
    elements=[]
    keepfiles=[]    

    for fname in files:

        try:
            found=re.search('\-(\w+)\.', fname).group(1)
        except AttributeError:
            print(f"WARNING: no element found in {fname}")
            found=''
        finally:
            if found in IGNORE_ELEMENTS:
                pass
            elif found in TRUE_ELEMENTS:
                elements.append(found)
                keepfiles.append(fname)
            else:
                pass
               # print(f"WARNING: Unexpected element {found} not used")

    files = keepfiles
    if len(elements) == len(files):
        zipped = zip(elements, files)    
        zipped_sorted = sorted(zipped)

        elements = [elements for elements, files  in zipped_sorted]
        files = [files for elements, files in zipped_sorted]

    else:
        raise ValueError("mismatch between elements and files")

    return elements, files


def load_maps(filepaths):
    print(filepaths)

    #load an image and check dimensions
    im = Image.open(filepaths[0])
    img = np.array(im)

    dims = img.shape

    maps=np.zeros((dims[0], dims[1], len(filepaths)), dtype=np.float32)

    i=0
    for f in filepaths:
            im = Image.open(f)
            img = np.array(im)
            #replace all negative values with 0
            img = np.where(img<0, 0, img)
            if not (img.shape == dims):
                raise ValueError(f"unexpected dimensions for file {f}")
            maps[:,:,i]=img
            i+=1

    print(f"Map shape: {maps.shape}")

    emptymin=0
    emptymax=0
    for i in range(maps.shape[0]):
        nmax=np.max(maps[i,:,:])
        navg=np.average(maps[i,:,:])
        #print(f"ROW {i}, max: {nmax}, avg: {navg}")
        if nmax == 0:
            
            if emptymin == 0:
                emptymin=i
                emptymax=i
                print(f"EMPTY ROW at {i}")
            elif emptymax == (i-1):
                emptymax = i
                print(f"EMPTY ROW at {i}")
            else:
                emptymax = i
                print(f"WARNING: DISCONTIGUOUS EMPTY ROW at {i}")

    maps=maps[0:emptymax,:,:]
    print(f"Revised map shape: {maps.shape}")
    data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
    print(f"Data shape: {data.shape}")

    dims=maps[:,:,0].shape
    #data=np.swapaxes(data,0,1)

    return data, dims

def modify_maps(data, elements):
    #BASEFACTOR=100000   #ppm to wt%
    BASEFACTOR=1/100000
    MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si']
    #MODIFY_FACTORS = [ 1000, 50, 20, 30 ]
    #MODIFY_FACTORS = [ 100, 5, 2, 3 ] <--best manual
    #MODIFY_FACTORS = [ 100, 5, 1, 1.5 ]
    MODIFY_FACTORS = [ 0.1, 0.1, 0.5, 1 ]

    """
    FUTURE: normalise MODIFY_LIST to MODIFY_SET eg. 1.0, 2.0, 3.0
    instead of using tuneable factor
    """


    #i=0
    #print(data.shape)
    #print(len(elements))
    #print(data.shape[1])

    for i in range(data.shape[1]):
        factor=BASEFACTOR

        #print(f"{elements[i]}, pre, max: {np.max(data[:,i])}")

        for idx, sname in enumerate(MODIFY_LIST):
            if elements[i] == sname:
                factor=MODIFY_FACTORS[idx]/np.max(data[:,i])
                #print(i, elements[i], idx, sname, factor)

        data[:,i]=(data[:,i]*factor)
        #print(f"{elements[i]}, post, max: {np.max(data[:,i])}, factor: {factor}")

    #    i+=1

    return data

def get_data(image_directory):

    print(image_directory)

    files = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files = get_elements(files)

    filepaths = [os.path.join(image_directory, file) for file in files ] 

    data, dims = load_maps(filepaths)

    print(elements)
    print(f"data shape: {data.shape}")
    #print(f"----{elements[8]} tracker: {np.max(data[:,8])}")    #DEBUG

    data = modify_maps(data, elements)

    #print(f"-----{elements[8]} tracker: {np.max(data[:,8])}")   #DEBUG

    #print(maps.shape, data.shape)

    return data, elements, dims

OVERWRITE=False

def process(data, dims, image_directory, force=False):

    if OVERWRITE:
        overwrite=force
    else:
        overwrite=False

    print(force, overwrite)

    categories, classavg, embedding, clusttimes = clustering.get(data, image_directory, force=force, overwrite=overwrite)

    return categories, classavg, embedding, clusttimes, data, dims


def plot_all(categories, classavg, embedding, data, elements, dims):

    IDX=8       #element index

    palette=processed_plots.build_palette(categories)

    processed_plots.show_map(data, elements, dims, IDX)

    processed_plots.category_map(categories, data, dims, palette=palette)
    
    processed_plots.category_avgs(categories, elements, classavg, palette=palette)

    processed_plots.seaborn_embedplot(embedding, categories, palette=palette)

#    processed_plots.seaborn_kdeplot(embedding, categories)

#   processed_plots.seaborn_kdecontours(embedding, categories)