import os
import argparse

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

    if args.index_only and args.fill_deadtimes:
        print("-------------------------------")
        print("WARNING: must parse map to use --fill-deadtimes")
        print("continuing with --index-only disabled")
        args.index_only = False

    if args.index_only and args.export_data:
        print("-------------------------------")
        print("WARNING: must parse map to use --export-data")
        print("continuing with --index-only disabled")
        args.index_only = False

    if args.input_file == None:   
        raise ValueError("No input file specified")

    if args.write_modified:
        if args.x_coords[1] == None:
            args.x_coords[1]=int(999999)
        if args.y_coords[1] == None:
            args.y_coords[1]=int(999999)

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
        "to export as csv, change SAVEFMT_READABLE = True in xfmreadout/config.yaml",
        action='store_true',
    )
    argparser.add_argument(
        "-w", "--write-modified", 
        help="Write modified .GeoPIXE file readable by CSIRO GeoPIXE package"
        "will crop to within --submap-coords"
        "will attempt to predict missing deadtimes from counts if --fill-deadtimes is specified.",
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
    #--------------------------
    #run control args
    argparser.add_argument(
        "-i", "--index-only", 
        help="Only index headers, do not extract data from files"
        "much faster but extracts header data only"
        "incompatible with --fill-deadtimes and --classify-spectra",
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
        "-dt", "--fill-deadtimes", 
        help="Predict deadtimes from counts"
        "use with ----write-modified"
        "not compatible with --index-only"
        "WARNING: experimental, prediction highly approximate",
        action='store_true',
    )
    argparser.add_argument(
        "-ff", "--force", 
        help="Force recalculation of all pixels/classes",
        action='store_true', 
    )

    #--------------------------
    #resource args eg. multiprocess, batch size
    argparser.add_argument(
        '-m', "--multiprocess", 
        help="Pre-cache memory using second process"
        "Prevents parse operation waiting on disk I/O"
        "Increases memory usage for buffer to 2x --memory-size",
        action='store_true', 
    )
    argparser.add_argument(
        '-s', "--chunk-size", 
        help="Size of memory buffer (in Mb) to load while parsing"
        "Defaults to 1000 (Mb)",
        type=int, 
        default=int(config['CHUNKSIZE']),
    )

    args = argparser.parse_args(args_in)

    args = checkargs(args, config)

    return args


def readargs_processed(args_in, config):
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
        "Results will be placed in a  ./outs/ subfolder within this directory"
        "Defaults to the directory containing the input file",
        type=os.path.abspath,
    )

    argparser.add_argument(
        "-ff", "--force", 
        help="Force recalculation of all pixels/classes",
        action='store_true', 
    )

    args = argparser.parse_args(args_in)

    return args