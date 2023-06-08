import time
import sys
import os
import numpy as np

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.colour as colour
import xfmreadout.clustering as clustering
import xfmreadout.visualisations as vis
import xfmreadout.structures as structures
import xfmreadout.dtops as dtops
import xfmreadout.parser as parser
import xfmreadout.diagops as diagops
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

    elements, data, dims, sd_data, sd_dims = processops.compile(image_directory)

    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")
    overwrite = ( args.force or args.force_clustering )
    categories, classavg, embedding, clusttimes, classifier = clustering.run(data, image_directory, force_embed=args.force, force_clust=args.force_clustering, overwrite=overwrite)
    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")

    vis.plot_clusters(categories, classavg, embedding, dims)

    for i in range(len(elements)):
        print(f"{elements[i]}, max: {np.max(data[:,i]):.2f}, 98: {np.quantile(data[:,i],0.98):.2f}, avg: {np.average(data[:,i]):.2f}")

if __name__ == '__main__':
 
    entry_processed()

    sys.exit()