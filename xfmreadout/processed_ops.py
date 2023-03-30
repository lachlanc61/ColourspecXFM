import os
import re
import sys
import numpy as np
import periodictable as pt
import matplotlib.pyplot as plt
from PIL import Image

import xfmreadout.clustering as clustering


IGNORE_ELEMENTS=['sum','Compton', 'Mo']

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




def modify_maps()
    BASEFACTOR=100000
    MODIFY_LIST = ['Na', 'Mg', 'Al', 'Si']
    MODIFY_FACTORS = [ 100, 1, 1, 1 ]




def main(image_directory):

    files = [f for f in os.listdir(wdir) if f.endswith('.tiff')]

    elements, files = get_elements(files)

    filepaths = [os.path.join(wdir, file) for file in files ] 

    #get file dimensions:

    im = Image.open(os.path.join(wdir, filepaths[0]))
    img = np.array(im)

    dims = img.shape

    maps=np.zeros((len(elements), dims[0], dims[1]), dtype=np.float32)


    i=0
    for f in filepaths:
        im = Image.open(f)
        img = np.array(im)
        #replace all negative values with 0
        img = np.where(img<0, 0, img)

        factor=BASEFACTOR
        for idx, snames in enumerate(MODIFY_LIST):
            if elements[i] in snames:
                factor=BASEFACTOR*MODIFY_FACTORS[idx]

        maps[i,:,:]=img/factor
        i+=1
        

    data=maps.reshape(maps.shape[0],-1)

    data=np.swapaxes(data,0,1)

    print(maps.shape, data.shape)
    


