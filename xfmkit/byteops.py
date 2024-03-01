import struct 

import logging
logger = logging.getLogger(__name__)

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def binunpack(stream, idx, sformat):
    """
    parse binary data via struct.unpack
    takes:
        stream of bytes
        byte index
        format flag for unpack (currently accepts: <H <f <I )
    returns:
        value in desired format (eg. int, float)
        next byte index
    """

    if sformat == "<H":
        nbytes=2
    elif sformat == "<f":
        nbytes=4
    elif sformat == "<I":
        nbytes=4
    else:
        raise ValueError(f"ERROR: {sformat} not recognised by local function binunpack")

    output = struct.unpack(sformat, stream[idx:idx+nbytes])[0]

    return(output)