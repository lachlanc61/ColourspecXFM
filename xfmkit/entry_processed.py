import sys
import os
import numpy as np

import logging
from logging.handlers import TimedRotatingFileHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xfmkit.config as config
import xfmkit.utils as utils
import xfmkit.argops as argops
import xfmkit.clustering as clustering
import xfmkit.visualisations as vis
import xfmkit.processops as processops
import xfmkit.structures as structures
import xfmkit.geopixeio as geopixeio

#-----------------------------------
#vars
#-----------------------------------
CONF_FILE_DEFAULT="conf/xfmkit.conf"

logger = logging.getLogger(__name__)

def logging_setup():
    logger.setLevel(logging.DEBUG)

    log_file = config.get('logging', 'log_file', default = "/var/log/xfmkit.log")

    filehandler = TimedRotatingFileHandler(log_file, when='midnight',backupCount=7)
    filehandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    filehandler.setFormatter(formatter)
    
    logger.addHandler(filehandler)

logging_setup()

def entry_processed(conf_file=CONF_FILE_DEFAULT):
    """
    entrypoint wrapper getting args from sys
    """
    args_in = sys.argv[1:]  #NB: exclude 0 == script name

    config.setup(conf_file=conf_file)

    read_processed(args_in)

    return


def read_processed(args_in):
    """
    read exported tiffs from geopixe
    
    perform clustering and visualisation
    """

    #get command line arguments
    args = argops.readargs_processed(args_in)

    image_directory=args.input_directory
    output_directory=os.path.join(image_directory, "analysis")

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    
    ds = processops.compile(image_directory)

    ds.crop((args.x_coords[0], args.x_coords[1]), (args.y_coords[0], args.y_coords[1]))

    pxs = structures.PixelSet(ds)

    pxs.downsample_by_se()

    pxs.apply_weights(amplify_list = args.amplify, 
                            suppress_list = args.suppress, 
                            ignore_list = args.ignore,
                            normalise = args.normalise, 
                            weight_transform = args.weight_transform, 
                            data_transform = args.data_transform 
                        )

    overwrite = ( args.force or args.force_clustering )

    categories, embedding, kde = clustering.run(pxs.weighted.d, output_directory, eom=args.classes_eom, majors=args.majors, target_components=args.n_components, force_embed=args.force, force_clust=args.force_clustering, overwrite=overwrite, do_kde=args.kde)

    classavg = clustering.get_classavg(pxs.data.d, categories, output_directory, labels=pxs.labels)

    weighted_avg = clustering.get_classavg(pxs.weighted.d, categories, output_directory, labels=pxs.labels)

    palette = vis.plot_clusters(categories, classavg, embedding, kde, pxs.data.dimensions, output_directory=output_directory, plot_kde=args.kde, labels=pxs.labels)

    geopixeio.export_regions(categories, pxs.dimensions, output_directory=output_directory)

    #vis.contours_3d(embedding)

    return pxs, embedding, categories, classavg, palette


if __name__ == '__main__':
 
    entry_processed()

    sys.exit()