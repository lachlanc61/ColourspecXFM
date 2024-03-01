import os
import argparse
import psutil
import logging

import xfmkit.processops as processops
import xfmkit.config as config

logger = logging.getLogger(__name__)


valid_weight_transforms=config.get('argparse', 'valid_weight_transforms')
default_weight_transform=config.get('argparse', 'default_weight_transform')
valid_data_transforms=config.get('argparse', 'valid_data_transforms')
ignore_lines=config.get('elements', 'ignore_lines')


def checkargs_processed(args):
    """
    sanity check on arg combinations
    """
    if args.x_coords == None:
        args.x_coords = [ 0, int(999999)]
    if args.y_coords == None:
        args.y_coords = [0 , int(999999)]

    if (args.x_coords[0] >= args.x_coords[1]):
        raise ValueError("First x_coordinate must be < second x_coordinate")
    if (args.y_coords[0] >= args.x_coords[1]):
        raise ValueError("First y_coordinate must be < second y_coordinate")

    if args.n_components <= 0 or args.n_components >= 100:
        raise ValueError("Invalid number of components, (expected 1 < n < 100)")

    if not args.weight_transform in valid_weight_transforms and not args.weight_transform is None:
        raise ValueError(f"Unrecognised weight transform {args.weight_transform}")

    if not args.data_transform in valid_data_transforms and not args.data_transform is None:
        raise ValueError(f"Unrecognised weight transform {args.data_transform}")

    processops.check_expected_lines(args.suppress)
    processops.check_expected_lines(args.suppress)

    return args

def readargs_processed(args_in):
    """
    read in a set of command-line args for analysing processed maps
    """

    #initialise the parser
    argparser = argparse.ArgumentParser(
        description="XFM data loader and analysis package"
    )

    #--------------------------
    #set up the expected args
    #--------------------------
    #inputs and outputs locations
    argparser.add_argument(
        "-d", "--input-directory", 
        help="Specify a directory containing processed .tiff files"
        "with pixel values corresponding to concentration, areal density, or counts",
        type=os.path.abspath,
    )
    argparser.add_argument(
        "-o", "--output-directory", 
        help="Specify the filepath to be used for outputs"
        "Results will be placed in a  ./outputs/ subfolder within this directory"
        "Defaults to the input directory",
        type=os.path.abspath,
    )
    argparser.add_argument(
        '-x', "--x-coords", 
        help="Start and end coordinates in X direction"
        "as: X_start, X_end"
        "Crop the exported map to these coordinates",
        nargs='+', 
        type=int, 
    )
    argparser.add_argument(
        '-y', "--y-coords", 
        help="Start and end coordinates in Y direction"
        "as: Y_start, Y_end"
        "Crop the exported map to these coordinates",
        nargs='+', 
        type=int, 
    )
    #----------------------------------------------------
    #classification options
    argparser.add_argument(
        "-ff", "--force", 
        help="Force recalculation of all pixels/classes",
        action='store_true', 
    )

    argparser.add_argument(
        "-fc", "--force-clustering", 
        help="Force recalculation of clusters - overridden by --force",
        action='store_true', 
    )   

    argparser.add_argument(
        "-som", "--use_som", 
        help="Classify via self-organising map",
        action='store_true', 
    )

    argparser.add_argument(
        '-n', "--n-components", 
        help="Number of components for reduction",
        type=int, 
        default=int(2)
    )

    argparser.add_argument(
        '-m', "--majors", 
        help="Cluster for majors only",
        action='store_true', 
    )

    argparser.add_argument(
        '-eom', "--classes_eom", 
        help="HDBSCAN setup: Use mass-based classification with default epsilon"
        "otherwise, estimate minimum size of clusters from number of pixels",
        action='store_true', 
    )    

    argparser.add_argument(
        "-k", "--kde", 
        help="Visualise kde",
        action='store_true', 
    )



    #----------------------------------------------------
    #pre-processing options

    argparser.add_argument(
        '-tn', "--normalise", 
        help="Normalise element images 0.0 <-> 1.0",
        action='store_true', 
    )

    argparser.add_argument(
        '-tw', "--weight_transform", 
        help="Transformation to apply to weights"
        f"recognised values: {valid_weight_transforms}",        
        type=str, 
        default=default_weight_transform
    )

    argparser.add_argument(
        '-td', "--data_transform", 
        help="Transformation to apply to data"
        f"recognised values: {valid_data_transforms}",        
        type=str, 
        default=None
    )

    argparser.add_argument(
        '-s', "--suppress", 
        help="Element/line symbols to be de-weighted",
        nargs='+', 
        type=str, 
        default=[],        
    )

    argparser.add_argument(
        '-i', "--ignore", 
        help="Element/line symbols to be ignored",
        nargs='+', 
        type=str, 
        default=ignore_lines,        
    )

    argparser.add_argument(
        '-a', "--amplify", 
        help="Element/line symbols to be up-weighted",
        nargs='+', 
        type=str, 
        default=[],
    )

    args = argparser.parse_args(args_in)

    args = checkargs_processed(args)

    return args