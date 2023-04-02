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
    
    data=maps.reshape(maps.shape[0],-1)

    data=np.swapaxes(data,0,1)

    return data, dims

def modify_maps(data, elements):
    BASEFACTOR=100000
    MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si']
    MODIFY_FACTORS = [ 100, 1, 1, 1 ]

    i=0
    for i in range(data.shape[1]):
        factor=BASEFACTOR

        for idx, snames in enumerate(MODIFY_LIST):
            if elements[i] in snames:
                factor=BASEFACTOR*MODIFY_FACTORS[idx]

        data[:,i]=data[:,i]/factor
        i+=1

    return data

def get_data(image_directory):

    files = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files = get_elements(files)

    filepaths = [os.path.join(image_directory, file) for file in files ] 

    data, dims = load_maps(filepaths)

    print(elements)
    print(data.shape)

    data = modify_maps(data, elements)

    #print(maps.shape, data.shape)

    return data, elements, dims

def process(data, dims, image_directory, force=False):

    print(force)

    categories, classavg, embedding, clusttimes = clustering.get(data, image_directory, force=force)

    return categories, classavg, embedding, clusttimes, data, dims


def plot_all(categories, classavg, embedding, data, elements, dims):

    IDX=5       #element index

    processed_plots.show_map(data, elements, dims, IDX)

    processed_plots.category_map(categories, data, dims)
    
    processed_plots.category_avgs(categories, elements, classavg)

    processed_plots.seaborn_embedplot(embedding, categories)

#    processed_plots.seaborn_kdeplot(embedding, categories)