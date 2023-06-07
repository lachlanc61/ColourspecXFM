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
            if found == "var":
                pass
            elif found in IGNORE_LINES:
                pass
            elif found in possible_lines:
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


def get_variance_files(elements, files):

    for fname in files:
        try:
            found=re.search('\-(\w+)-var\.', fname).group(1)
            print(f"var: {found}")
        except AttributeError:
            found=''
    return 

def maps_load(filepaths):
    """
    load maps from datafiles, cleanup and reshape

    return dataset and dimensions
    """    

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

    return maps

def maps_unroll(maps):
    """
    reshape map (x, y, counts) to data (i, counts)

    returns dataset and dimensions
    """

    data=maps.reshape(maps.shape[0]*maps.shape[1],-1)

    print(f"Final shape: {data.shape}")

    dims=maps[:,:,0].shape

    return data, dims

def maps_cleanup(maps):
    """
    discard empty rows at end of map
    """

    print(f"Initial shape: {maps.shape}")

    empty_begin=0
    empty_last=0
    for i in range(maps.shape[0]):
        row_maxcounts=np.max(maps[i,:,:])
        if row_maxcounts == 0:
            if empty_begin == 0:
                empty_begin=i
                empty_last=i
                print(f"WARNING: found empty row at {i} of {maps.shape[0]-1}")
            elif empty_last == (i-1):
                empty_last = i
            else:
                empty_last = i
                print(f"WARNING: DISCONTIGUOUS EMPTY ROW at {i}")

    maps=maps[0:empty_begin,:,:]
    print(f"Revised shape: {maps.shape}")

    return maps


def maps_crop(maps, x_min=0, x_max=9999, y_min=0, y_max=9999):
    """
    crop map to designated size
    """     
    maps=maps[y_min:y_max,x_min:x_max,:]        #will likely fail if default out of range

    dims=maps[:,:,0].shape

    print(f"Cropped shape: {maps.shape}")

    return maps


def data_normalise(data, elements):

    #iterate through all elements
    for i in range(data.shape[1]):
        factor=BASEFACTOR

        #check if element in MODIFY_LIST
        #   then norm to MODIFY_FACTOR
        for idx, sname in enumerate(MODIFY_LIST):
            if elements[i] == sname:
                factor=MODIFY_NORMS[idx]/np.max(data[:,i])
                print(f"--- scaling {sname} to {MODIFY_NORMS[idx]}")

        data[:,i]=(data[:,i]*factor)

    return data

def compile(image_directory, x_min=0, x_max=9999, y_min=0, y_max=9999):
    """
    read tiffs from image directory 
    
    return corrected 2D stack, array of elements, and dimensions
    """

    print("-----------------")
    print("BEGIN reading processed data")
    print(f"Location: {image_directory}")
    print("-----")

    files_all = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files_maps = get_elements(files_all)

    files_variance = get_variance_files(elements, files_all)
    #read variance
    #repeat below for var
    #need to check for element map w/o variance and create dummy
    
    filepaths = [os.path.join(image_directory, file) for file in files_maps ] 

    maps = maps_load(filepaths)

    maps = maps_cleanup(maps)

    maps = maps_crop(maps, x_min, x_max, y_min, y_max)

    data, dims = maps_unroll(maps)

    print("-----")
    print("Elements identified:")
    print(elements)

    data = data_normalise(data, elements)

    print("-----------------")

    return data, elements, dims
