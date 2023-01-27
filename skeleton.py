import numpy as np



"""
xfmap object
static:
    dimensions, resolution -> everything pulled from header
    fullsize, chunksize

    infile to read
dynamic:
    stream
    streamlen
    
    idx     #byte pointer
    pxidx   #pixel pointer
    rowidx  #row pointer
    pxstart #pixel origin pointer

"""




def getindexes(start_index, xfmap, npx):
    pixelindeces=np.zeros(npx, dtype=np.int64)

    pass
    return pixelindeces

def readpixelheader(pixel_index, xfmap):
    pass
    return pxlen, xidx, yidx, det, dt


def readpixeldata(pixel_index, xfmap):
    pass
    return chan, counts




"""
current 

    up to end of xfmap init
        (at getdetectors)

    think we don't need getdetectors anymore

    can index all pixels and do this afterwards, I think

"""