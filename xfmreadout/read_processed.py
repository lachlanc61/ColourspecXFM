import time
import sys
import os
import numpy as np

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.processops as processops
import xfmreadout.clustering as clustering
import xfmreadout.visualisations as vis

#-----------------------------------
#vars
#-----------------------------------
PACKAGE_CONFIG='xfmreadout/config.yaml'

def entry():
    """
    entry point without explicit args
    """
    args_in = sys.argv[1:]  #NB: exclude 0 == script name
    main(args_in)

def main(args_in):

    #create input config from args and config files
    config =utils.initcfg(PACKAGE_CONFIG)

    #get command line arguments
    args = argops.readargs_processed(args_in, config)

    image_directory=args.input_directory
    output_directory=os.path.join(image_directory, "outputs")

    data, elements, dims = processops.compile(image_directory)

    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")
    categories, classavg, embedding, clusttimes = clustering.run(data, image_directory)
    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")

    vis.plot_clusters(categories, classavg, embedding, dims)

    for i in range(len(elements)):
        print(f"{elements[i]}, max: {np.max(data[:,i]):.2f}, 98: {np.quantile(data[:,i],0.98):.2f}, avg: {np.average(data[:,i]):.2f}")


if __name__ == "__main__":
    args_in = sys.argv[1:]

    main(args_in)      
    sys.exit()