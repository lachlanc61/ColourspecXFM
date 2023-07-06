import time
import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.clustering as clustering
import xfmreadout.visualisations as vis
import xfmreadout.processops as processops

"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- extracts pixel data
- classifies data via eg. UMAP, HDBSCAN
- displays classified maps
- produces average spectrum per class

./data has example datasets
"""
#-----------------------------------
#vars
#-----------------------------------
PACKAGE_CONFIG='xfmreadout/config.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

def entry_processed():
    """
    entrypoint wrapper getting args from sys
    """
    args_in = sys.argv[1:]  #NB: exclude 0 == script name
    read_processed(args_in)

    return

def read_processed(args_in):
    """
    read exported tiffs from geopixe
    
    perform clustering and visualisation
    """
    #create input config from args and config files
    config =utils.initcfg(PACKAGE_CONFIG)

    #get command line arguments
    args = argops.readargs_processed(args_in, config)

    image_directory=args.input_directory
    output_directory=os.path.join(image_directory, "outputs")

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    
    elements, data, dims, sd_data, sd_dims = processops.compile(image_directory)

    data, dims = processops.data_crop(data, dims, args.x_coords[0], args.x_coords[1], args.y_coords[0], args.y_coords[1])

    overwrite = ( args.force or args.force_clustering )
    categories, classavg, embedding = clustering.run(data, image_directory, sqrt=True, force_embed=args.force, force_clust=args.force_clustering, overwrite=overwrite)

    vis.plot_clusters(categories, classavg, embedding, dims, output_directory=output_directory)

    concentration_averages = pd.DataFrame(data=classavg, columns=elements)

    print(tabulate(concentration_averages, headers='keys', tablefmt='psql'))

    return

if __name__ == '__main__':
 
    entry_processed()

    sys.exit()