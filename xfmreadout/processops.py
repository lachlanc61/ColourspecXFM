import os
import re
import numpy as np
import periodictable as pt
from PIL import Image

import xfmreadout.clustering as clustering


FORCE = True
AUTOSAVE = True

EMBED_DIRNAME = "embedding"

#IGNORE_LINES=['sum','Back','Compton','Mo','MoL']
IGNORE_LINES=['Ar']
CUSTOM_LINES=['sum','Back','Compton']
Z_CUTOFFS=[11, 55, 37, 73]       #K min, K max, L min, M min

MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si', 'Cl', 'sum', 'Back', 'Mo', 'MoL', 'Compton', 'S']
MODIFY_NORMS = [ 0.005, 0.01, 0.025, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0 ]
BASEFACTOR=1/100000 #ppm to wt%

def get_elements(files):
    """

    Extract element names and corresponding files

    Ignore files that do not correspond to elements

    """
    elements=[]
    possible_lines = []
    keepfiles=[]    

    #use the periodic table and known z-cutoffs to get possible lines
    for ptelement in pt.elements:
        #add three versions of the line: K (unlabelled), L, M
        if ptelement.number >= Z_CUTOFFS[0] and ptelement.number <= Z_CUTOFFS[1]:
            possible_lines.append(ptelement.symbol)
        if ptelement.number >= Z_CUTOFFS[2]:
            possible_lines.append(ptelement.symbol+"L")
        if ptelement.number >= Z_CUTOFFS[3]:            
            possible_lines.append(ptelement.symbol+"M")

    for line in CUSTOM_LINES:
        possible_lines.append(line)

    for fname in files:

        try:
            found=re.search('\-(\w+)\.', fname).group(1)
        except AttributeError:
            print(f"WARNING: no element found in {fname}")
            found=''
        finally:
            if found in IGNORE_LINES:
                pass
            elif found in possible_lines:
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


def load_maps(filepaths, x_min=0, x_max=9999, y_min=0, y_max=9999):
    
    if False:
        print(f"WARNING: MANUAL CROP ACTIVE")
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

    maps=maps[y_min:y_max,x_min:x_max,:]
    print(f"Revised map shape: {maps.shape}")
    data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
    print(f"Data shape: {data.shape}")

    dims=maps[:,:,0].shape
    #data=np.swapaxes(data,0,1)

    return data, dims

def modify_maps(data, elements):

    #iterate through all elements
    for i in range(data.shape[1]):
        factor=BASEFACTOR

        #check if element in MODIFY_LIST
        #   then norm to MODIFY_FACTOR
        for idx, sname in enumerate(MODIFY_LIST):
            if elements[i] == sname:
                factor=MODIFY_NORMS[idx]/np.max(data[:,i])

        data[:,i]=(data[:,i]*factor)

    return data

def compile(image_directory, x_min=0, x_max=9999, y_min=0, y_max=9999):

    print(image_directory)

    files = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files = get_elements(files)

    filepaths = [os.path.join(image_directory, file) for file in files ] 

    data, dims = load_maps(filepaths, x_min, x_max, y_min, y_max)

    print(elements)
    print(f"data shape: {data.shape}")
    #print(f"----{elements[8]} tracker: {np.max(data[:,8])}")    #DEBUG

    data = modify_maps(data, elements)

    #print(f"-----{elements[8]} tracker: {np.max(data[:,8])}")   #DEBUG

    #print(maps.shape, data.shape)

    return data, elements, dims
