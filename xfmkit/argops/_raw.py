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

def checkargs(args, config):
    """
    sanity check on arg combinations
    - warns and adjusts args on soft conflicts
    - flags and raises on hard conflicts
    - corrects units for eg. chunksize
    """

    if args.index_only and args.classify_spectra:
        print("-------------------------------")
        print("WARNING: must parse map to use --classify-spectra")
        print("continuing with --index-only disabled")
        args.index_only = False

    if args.index_only and args.modify_deadtimes < 0:
        print("-------------------------------")
        print("WARNING: must parse map to predict deadtimes")
        print("continuing with --index-only disabled")
        args.index_only = False

    if not args.modify_deadtimes > 100 and not args.write_modified:
        print("-------------------------------")
        print("WARNING: must write map to apply modified deadtimes")
        print("continuing with --write-modified enabled")
        args.write_modified = True

    if args.index_only and args.export_data:
        print("-------------------------------")
        print("WARNING: must parse map to use --export-data")
        print("continuing with --index-only disabled")
        args.index_only = False

    if args.input_file == None:   
        raise ValueError("No input file specified")

    if args.modify_deadtimes >= 0 and args.modify_deadtimes <= 100:
        #we are assigning fixed DT
        pass
    elif args.modify_deadtimes < 0:
        #we are predicting DT
        pass
    elif args.modify_deadtimes > 100:
        #we are not changing DT
        pass
    else:
       raise ValueError("modify-deadtimes value out of range")

    if args.write_modified:
        if args.x_coords == None:
            args.x_coords = [ 0, int(999999)]
        if args.y_coords == None:
            args.y_coords = [0 , int(999999)]

        if (args.x_coords[0] >= args.x_coords[1]):
            raise ValueError("First x_coordinate must be < second x_coordinate")
        if (args.y_coords[0] >= args.x_coords[1]):
            raise ValueError("First y_coordinate must be < second y_coordinate")

    elif args.x_coords != None or args.y_coords != None:
        print("-------------------------------")
        print("WARNING: crop coordinates given without --write-modified")
        print("cropped .GeoPIXE file will not will be produced")        

    #if chunk size is small, convert to bytes
    if args.chunk_size < config['MBCONV']:
        args.chunk_size=args.chunk_size*config['MBCONV']

    return args


def readargs(args_in, config):
    """
    read in a set of command-line args and return the parsed object

    reads some fallback defaults from config (eg. chunk size)
    """

    #initialise the parser
    argparser = argparse.ArgumentParser(
        description="XFM data loader and analysis package"
    )

    #--------------------------
    #set defaults where needed
    #--------------------------

    #attempt to guess chunksize from memory via psutils
    vmem=psutil.virtual_memory()

    default_chunksize=round(vmem[0]*float(config['CHUNK_FRACTION'])*1e-6)

        #sanity check and use config if failed
    if not isinstance(default_chunksize, int) or default_chunksize <= 1000:
        default_chunksize = config['CHUNKSIZE']

    #--------------------------
    #set up the expected args
    #--------------------------
    #inputs and outputs locations
    argparser.add_argument(
        "-f", "--input-file", 
        help="Specify a .GeoPIXE file to be read in", 
        type=os.path.abspath,
    )
    argparser.add_argument(
        "-o", "--output-directory", 
        help="Specify the filepath to be used for outputs"
        "Results will be placed in a  ./outs/ subfolder within this directory"
        "Defaults to the directory containing the input file",
        type=os.path.abspath,
    )
    argparser.add_argument(
        "-l", "--log-file", 
        help="Specify a log file to be read in",
        type=os.path.abspath,
    )
    #--------------------------
    #write control
    argparser.add_argument(
        "-e", "--export-data", 
        help="Export pixel data to .npy file"
        "will extract spectrum-per-pixel-per-detector data to .npy file-like object"
        "file can be opened with numpy.load(filepath)"
        "to export as csv, change SAVEFMT_READABLE = True in xfmkit/config.yaml",
        action='store_true',
    )
    argparser.add_argument(
        "-w", "--write-modified", 
        help="Write modified .GeoPIXE file readable by CSIRO GeoPIXE package"
        "will crop to within --submap-coords"
        "will attempt to predict missing deadtimes from counts if --modify-deadtimes is specified.",
        action='store_true',
    )
    argparser.add_argument(
        '-x', "--x-coords", 
        help="Start and end coordinates in X direction"
        "as: X_start, X_end"
        "Will crop exported .GeoPIXE file to within these coordinates"
        "Does not affect parsing and analysis, only exported .geoPIXE file"
        "use with --write-modified",
        nargs='+', 
        type=int, 
    )
    argparser.add_argument(
        '-y', "--y-coords", 
        help="Start and end coordinates in Y direction"
        "as: Y_start, Y_end"
        "Will crop exported .GeoPIXE file to within these coordinates"
        "Does not affect parsing and analysis, only exported .geoPIXE file"
        "use with --write-modified",
        nargs='+', 
        type=int, 
    )
    argparser.add_argument(
        "-dt", "--modify-deadtimes", 
        help="Fill or predict deadtimes and write to output .GeoPIXE file"
        "If given a float from 0-100, will fill that value"
        "If no value is given, will perform a per-pixel prediction from parsed counts"
        "Use with ----write-modified"
        "WARNING: experimental, prediction highly approximate",
        const=float(-1),
        default=float(101),
        action='store',
        nargs='?',
        type=float,
        #DEFAULTS: 
        #   if arg not given, value = 101 
        #       = do not fill deadtimes
        #   if arg given but no value specified, value = -1
        #        = predict deadtimes from counts

    )
    #--------------------------
    #run control args
    argparser.add_argument(
        "-i", "--index-only", 
        help="Only index headers, do not extract data from files"
        "much faster but extracts header data only"
        "incompatible with --modify-deadtimes and --classify-spectra",
        action='store_true',
    )
    argparser.add_argument(
        "-a", "--analyse", 
        help="Perform analysis operations"
        "generate deadtime maps and RGB spectral representation",
        action='store_true',
    )
    argparser.add_argument(
        "-c", "--classify-spectra", 
        help="Perform clustering operations"
        "generate spectral classification maps" 
        "uses PCA, UMAP and k-means to produce clusters based on raw spectra"
        "not compatible with --index-only",
        action='store_true',
    )
    argparser.add_argument(
        "-ff", "--force", 
        help="Force recalculation of all pixels/classes",
        action='store_true', 
    )

    #--------------------------
    #resource args eg. multiload, batch size
    argparser.add_argument(
        '-m', "--multiload", 
        help="Pre-cache memory using second process"
        "Prevents parse operation waiting on disk I/O"
        "Increases memory usage for buffer to 2x --memory-size",
        action='store_true', 
    )
    argparser.add_argument(
        '-p', "--python-only", 
        help="Parse using python only"
        "Exclude C++ submodule",
        action='store_true', 
    )    
    argparser.add_argument(
        '-s', "--chunk-size", 
        help="Size of memory buffer (in Mb) to load while parsing"
        "Defaults to 1000 (Mb)",
        type=int, 
        default=int(default_chunksize),
    )

    args = argparser.parse_args(args_in)

    args = checkargs(args, config)

    return args