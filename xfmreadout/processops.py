import os
import re
import numpy as np
import periodictable as pt
from PIL import Image

import xfmreadout.clustering as clustering


FORCE = True
AUTOSAVE = True

EMBED_DIRNAME = "embedding"

IGNORE_ELEMENTS=['sum','Back','Compton','Mo','MoL']


def get_elements(files):
    """

    Extract element names and corresponding files

    Ignore files that do not correspond to elements

    """
    elements=[]
    true_elements = []
    keepfiles=[]    

    for ptelement in pt.elements:
        true_elements.append(ptelement.symbol)

    for fname in files:

        try:
            found=re.search('\-(\w+)\.', fname).group(1)
        except AttributeError:
            print(f"WARNING: no element found in {fname}")
            found=''
        finally:
            if found in IGNORE_ELEMENTS:
                pass
            elif found in true_elements:
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

    #YMIN=0
    #YMAX=9999
    #XMIN=0
    #XMAX=9999
    YMIN=100
    YMAX=275
    XMIN=50
    XMAX=600

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



    maps=maps[YMIN:YMAX,XMIN:XMAX,:]
    print(f"Revised map shape: {maps.shape}")
    data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
    print(f"Data shape: {data.shape}")

    dims=maps[:,:,0].shape
    #data=np.swapaxes(data,0,1)

    return data, dims

def modify_maps(data, elements):
    #BASEFACTOR=100000   #ppm to wt%
    BASEFACTOR=1/100000
    MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si', 'Cl']
    #MODIFY_FACTORS = [ 1000, 50, 20, 30 ]
    #MODIFY_FACTORS = [ 100, 5, 2, 3 ] <--best manual
    #MODIFY_FACTORS = [ 100, 5, 1, 1.5 ]
    MODIFY_FACTORS = [ 0.1, 0.1, 0.5, 1, 1]

    #iterate through all elements
    for i in range(data.shape[1]):
        factor=BASEFACTOR

        #check if element in MODIFY_LIST
        #   then norm to MODIFY_FACTOR
        for idx, sname in enumerate(MODIFY_LIST):
            if elements[i] == sname:
                factor=MODIFY_FACTORS[idx]/np.max(data[:,i])

        data[:,i]=(data[:,i]*factor)

    return data

def compile(image_directory):

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
