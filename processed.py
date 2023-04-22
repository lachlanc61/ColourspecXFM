import time
import sys
import os
import numpy as np

import xfmreadout.utils as utils
import xfmreadout.argops as argops
import xfmreadout.processops as processops

#-----------------------------------
#vars
#-----------------------------------
PACKAGE_CONFIG='xfmreadout/config.yaml'

def main(args_in):
    #create input config from args and config files
    config =utils.initcfg(PACKAGE_CONFIG)

    #get command line arguments
    args = argops.readargs_processed(args_in, config)

    image_directory=args.input_directory
    output_directory=os.path.join(image_directory, "outputs")

    data, elements, dims = processops.get_data(image_directory)

    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")

    categories, classavg, embedding, clusttimes, data, dims = processops.process(data, dims, image_directory, force=args.force)
    print(f"-----{elements[10]} tracker: {np.max(data[:,10])}")

    for i in range(len(elements)):
        print(f"{elements[i]}, {np.max(data[:,i])}")

    processops.plot_all(categories, classavg, embedding, data, elements, dims)


if __name__ == "__main__":
    main(sys.argv[1:])      #NB: exclude 0 == script name

    sys.exit()