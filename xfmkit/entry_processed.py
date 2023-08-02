import sys
import os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xfmkit.utils as utils
import xfmkit.argops as argops
import xfmkit.clustering as clustering
import xfmkit.visualisations as vis
import xfmkit.processops as processops
import xfmkit.structures as structures

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
PACKAGE_CONFIG='xfmkit/config.yaml'

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
    
    ds = processops.compile(image_directory)

    ds.crop((args.x_coords[0], args.x_coords[1]), (args.y_coords[0], args.y_coords[1]))

    pxs = structures.PixelSet(ds)

    pxs.modify_weights(do_sqrt=False)

    pxs.apply_weights()
    pxs.weighted.set_to(np.sqrt(pxs.weighted.d))

    overwrite = ( args.force or args.force_clustering )

    categories, embedding, kde = clustering.run(pxs.weighted.d, image_directory, target_components=args.n_components, force_embed=args.force, force_clust=args.force_clustering, overwrite=overwrite)

    classavg = clustering.get_classavg(pxs.data.d, categories, image_directory, force=args.force_clustering, overwrite=overwrite)

    palette = vis.plot_clusters(categories, classavg, embedding, kde, pxs.data.dimensions, output_directory=output_directory, plot_kde=args.kde)

    vis.table_classavg(classavg, pxs.labels)

    #vis.contours_3d(embedding)

    return pxs, embedding, categories, classavg, palette


if __name__ == '__main__':
 
    entry_processed()

    sys.exit()