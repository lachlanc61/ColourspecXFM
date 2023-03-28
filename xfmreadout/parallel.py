import numpy as np
#import multiprocessing as mp
import time

from multiprocessing import Pool, RawArray

import xfmreadout.bufferops as bufferops
import xfmreadout.utils as utils


# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def worker_func():
    pass
    """
    stream = buffer.data[relidx+pxheaderlen:relidx+pxlength]

    chan, counts = bufferops.readpxdata(stream, len(stream), bytesperchan, nchannels)
    """



def worker(bufferdata_in, indices_in, pxlens_in, pxheaderlen: int, bytesperchan: int, nchannels: int):
    """

    working from:
    https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

    alternate using manager:
    https://stackoverflow.com/questions/70352530/sharing-bytearray-between-processes-with-multiprocessing-manager

    
    ok, problem that <bytes> object doesn't seem straightforward to pass between workers
    no obvious object - gets passed as c_byte_array that seems to come out as ints when sliced 
    - may be possible to use manager via above link to create fully custom proxy class, but looks pretty complex
    - also not clear how manager interacts with pool

    also no guarantee of any performance gain as there is a lot of overhead for python multiprocessing

    maybe this is too-hard basket for now...

    """
    
    bufferdata_raw = RawArray('b', len(bufferdata_in) )
    indices_raw = RawArray('d', len(indices_in) )
    pxlens_raw = RawArray('d', len(pxlens_in) )

    #bufferdata = np.frombuffer(bufferdata_raw)
    indices = np.frombuffer(indices_raw)
    pxlens = np.frombuffer(pxlens_raw)

    #np.copyto(bufferdata, bufferdata_in)
    np.copyto(indices, indices_in)
    np.copyto(pxlens, pxlens_in)

    with Pool(processes=4, initializer=init_worker, initargs=(bufferdata_raw,indices_raw,pxlens_raw)) as pool:
        pass




    a=2

    pass
