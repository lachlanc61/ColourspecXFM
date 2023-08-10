import os
import re
import numpy as np
import periodictable as pt
from PIL import Image
from math import sqrt

import xfmkit.utils as utils
import xfmkit.structures as structures
import xfmkit.config as config

import logging
logger = logging.getLogger(__name__)

IGNORE_LINES=[]
AFFECTED_LINES=['Ar', 'Mo', 'MoL']
NON_ELEMENT_LINES=['sum','Back','Compton']

Z_CUTOFFS=[11, 55, 37, 73]  # cutoffs for K, L, M lines 
                            #   K min, K max, L min, M min

BASEFACTOR=1/10000 #ppm to wt%

def get_possible_lines():
    """
    #use the periodic table and known z-cutoffs to get possible lines
    """
    
    possible_lines = []

    for ptelement in pt.elements:
        #add three versions of the line: K (unlabelled), L, M
        if ptelement.number >= Z_CUTOFFS[0] and ptelement.number <= Z_CUTOFFS[1]:
            possible_lines.append(ptelement.symbol)
        if ptelement.number >= Z_CUTOFFS[2]:
            possible_lines.append(ptelement.symbol+"L")
        if ptelement.number >= Z_CUTOFFS[3]:            
            possible_lines.append(ptelement.symbol+"M")

    for line in NON_ELEMENT_LINES:
        possible_lines.append(line)
    
    return possible_lines


def check_expected_lines(lines):
    """
    checks that a list of line strings are all expected elements/lines
    """
    for line in lines:
        if not line in possible_lines:
            raise ValueError(f"Unexpected line {line}, exiting")

    return True

def get_elements(files):
    """
    Extract element names and corresponding files

    Ignore files that do not correspond to elements

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
    """
    search for variance sidecar files in filelist

    return list of variance files, with order matching elements

    raise an error if one is missing
    """

    variance_files = []

    variance_present = False

    for fname in files:
        try:
            symbol=re.search('\-(\w+)-var\.', fname).group(1)
        except AttributeError:
            symbol=''
        if symbol != '':
            variance_present = True
            break

    if variance_present:

        for element in elements:

            found_variance = False

            for fname in files:
                try:
                    symbol=re.search('\-(\w+)-var\.', fname).group(1)
                except AttributeError:
                    symbol=''

                if symbol == element:
                    variance_files.append(fname)
                    found_variance = True
                    break

            if found_variance == False:
                raise FileNotFoundError(f"No variance found for element {element}")

    return variance_files

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

    print(f"Initial shape: {maps.shape}")

    return maps

def maps_cleanup(maps):
    """
    discard empty rows at end of map
    """
    EMPTY_DEFAULT=99999
    empty_begin=EMPTY_DEFAULT
    empty_end=EMPTY_DEFAULT
    for i in range(maps.shape[0]):
        row_maxcounts=np.max(maps[i,:,:])
        if row_maxcounts == 0:
            if empty_begin == EMPTY_DEFAULT:
                empty_begin=i
                empty_end=i
                print(f"WARNING: found empty row at {i} of {maps.shape[0]-1}")
            elif empty_end == (i-1):
                empty_end = i
            else:
                empty_end = i
                print(f"WARNING: DISCONTIGUOUS EMPTY ROW at {i}")
        else:
            #if maxcount > 0 but empty_begin has been assigned, reset both empties
            if not empty_begin == EMPTY_DEFAULT:
                print(f"WARNING: block of data after empty rows from {empty_begin} to {empty_end}, resetting")
                empty_begin=EMPTY_DEFAULT
                empty_end=EMPTY_DEFAULT

    maps=maps[0:empty_begin,:,:]
    print(f"Revised shape: {maps.shape}")

    return maps


def printqvals(data, element, qval):
        avg_data = np.average(data)
        max_data = np.max(data)
        qmean = utils.mean_within_quantile(data, qmin=qval)
        qnum = np.quantile(data,qval)

        print(f"{element} -- data: {avg_data:.3f}, data(max): {max_data:.3f}, qmean: {qmean:.3f}, qnum: {qnum:.3f}")

        return

def printsdvals(data, element, qval):
        avg_data = np.average(data)
        max_data = np.max(data)
        qmean = utils.mean_within_quantile(data, qmin=qval-0.25, qmax=qval)
        qnum = np.quantile(data,qval)

        print(f"{element} -- sd: {avg_data:.3f}, sd(max): {max_data:.3f}, qmean: {qmean:.3f}, qnum: {qnum:.3f}")

        return


def variance_to_std(data):
    """
    convert variance stats to standard deviations via sqrt
    """
    result = np.sqrt(data)
    return result

def ppm_to_wt(data):
    """
    convert from ppm (as-read) to wt% 
    """
    result = data*BASEFACTOR
    return result

def extract_data(image_directory, files, is_variance=False):
    """
    get data from list of tiffs

    kwarg to specify whether reading variance or maps
    """

    filepaths = [os.path.join(image_directory, file) for file in files ] 

    maps = maps_load(filepaths)

    maps = maps_cleanup(maps)

    data, dims = utils.map_unroll(maps)

    print("-----")

    return data, dims

def compile(image_directory):
    """
    read tiffs from GeoPIXE image directory 
    
    return corrected 2D stack, array of elements, and dimensions
    """

    print("-----------------")
    print("BEGIN reading processed data")
    print(f"Location: {image_directory}")
    print("-----")

    files_all = [f for f in os.listdir(image_directory) if f.endswith('.tiff')]

    elements, files_maps = get_elements(files_all)

    check_expected_lines(elements)

    files_variance = get_variance_files(elements, files_all)
    
    if files_variance != []:
        variance_found = True

    print(f"Map files found: {len(files_maps)}")
    print(f"Elements identified: {elements}")

    if len(files_maps) != len(files_variance):
        raise ValueError("Mismatch between map and variance files")

    print("-----------------")    
    print(f"READING MAP DATA")
    data, dims = extract_data(image_directory, files_maps)

    dataseries = structures.DataSeries(data, dims)

    if variance_found:
        print("-----------------")
        print(f"READING VARIANCE DATA")
        var_data, se_dims = extract_data(image_directory, files_variance, is_variance=True)
        se_data = variance_to_std(var_data)
        seseries = structures.DataSeries(se_data, se_dims)

        ds = structures.DataSet(dataseries, se=seseries, labels=elements)

        ds.downsample_by_se()

    else:
        ds = structures.DataSet(dataseries, labels=elements)

    print("-----------------")
    print(f"Final shape: {ds.data.shape}")

    print("-----------------")
    print(f"Element values:")    
    for i in range(len(elements)):
        print(f"{elements[i]}, max: {np.max(ds.data.d[:,i]):.2f}, 98: {np.quantile(ds.data.d[:,i],0.98):.2f}, avg: {np.average(ds.data.d[:,i]):.2f}")
    print("-----------------")

    return ds


#module local variables

possible_lines = get_possible_lines()