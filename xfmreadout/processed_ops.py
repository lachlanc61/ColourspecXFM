import os
import re
import numpy as np
import periodictable as pt
import pandas as pd
from PIL import Image

import xfmreadout.clustering as clustering
import xfmreadout.processed_plots as processed_plots


FORCE = False
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
                print(f"WARNING: Unexpected element {found} not used")

    files = keepfiles
    if len(elements) == len(files):
        zipped = zip(elements, files)    
        zipped_sorted = sorted(zipped)

        elements = [elements for elements, files  in zipped_sorted]
        files = [files for elements, files in zipped_sorted]

    else:
        raise ValueError("mismatch between elements and files")

    return elements, files


def modify_maps(maps, elements):
    BASEFACTOR=100000
    MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si']
    MODIFY_FACTORS = [ 100, 1, 1, 1 ]

    i=0
    for i in range(maps.shape[0]):
        factor=BASEFACTOR

        for idx, snames in enumerate(MODIFY_LIST):
            if elements[i] in snames:
                factor=BASEFACTOR*MODIFY_FACTORS[idx]

        maps[i,:,:]=maps[i,:,:]/factor
        i+=1

    return maps


def load_maps(filepaths):

    #load an image and check dimensions
    im = Image.open(filepaths[0])
    img = np.array(im)

    dims = img.shape

    maps=np.zeros((len(filepaths), dims[0], dims[1]), dtype=np.float32)

    i=0
    for f in filepaths:
            im = Image.open(f)
            img = np.array(im)
            #replace all negative values with 0
            img = np.where(img<0, 0, img)
            if not (img.shape == dims):
                raise ValueError(f"unexpected dimensions for file {f}")
            maps[i,:,:]=img
            i+=1
    
    return maps


def get_embedding(data, mapshape, image_directory):
    """
    calculate embedding
    """

    N_CLUSTERS=6

    EMBED_DIR=os.path.join(image_directory,EMBED_DIRNAME)

    if not os.path.exists(EMBED_DIR):
        os.mkdir(EMBED_DIR)

    NPX=mapshape[1]*mapshape[2]

    NCHAN=mapshape[0]

    file_cats=os.path.join(EMBED_DIR,"categories.npy")
    file_classes=os.path.join(EMBED_DIR,"classavg.npy")
    file_embed=os.path.join(EMBED_DIR,"embedding.npy")
    file_ctime=os.path.join(EMBED_DIR,"clusttimes.npy")

    filesexist = os.path.isfile(file_cats) and os.path.isfile(file_classes) \
        and  os.path.isfile(file_embed) and os.path.isfile(file_ctime)

    if FORCE or not filesexist:
        categories, classavg, embedding, clusttimes = clustering.calculate(data, NPX, N_CLUSTERS, NCHAN )
        #embedding, clusttimes = clustering.reduce(data)
        if AUTOSAVE:
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

def plot_all(categories, classavg, embedding, maps, elements):

    IDX=5       #element index
    REDUCER=1   #reducer to use

    processed_plots.show_map(maps, elements, IDX)

    processed_plots.category_map(categories, maps)
    
    processed_plots.category_avgs(categories, elements, classavg)


    df= pd.DataFrame(embedding[REDUCER,:], columns=["x","y"])

    df["cat"]=categories[REDUCER,:]

    processed_plots.seaborn_embedplot(df)
    processed_plots.seaborn_kdeplot(df)


def main(image_directory):

    files = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files = get_elements(files)

    filepaths = [os.path.join(image_directory, file) for file in files ] 

    maps = load_maps(filepaths)

    maps = modify_maps(maps, elements)

    data=maps.reshape(maps.shape[0],-1)

    data=np.swapaxes(data,0,1)

    #print(maps.shape, data.shape)

    categories, classavg, embedding, clusttimes = get_embedding(data, maps.shape, image_directory)

    plot_all(categories, classavg, embedding, maps, elements)

    return categories, classavg, embedding, clusttimes, maps, elements

