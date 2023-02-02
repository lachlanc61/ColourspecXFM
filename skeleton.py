import numpy as np





"""
modularise - refactoring config to use flags/kwargs

-i index 
    -> save pixel header data
-p parse 
    -> save pixel header data + sums, flatsums
    >> -i
-w write
    -> write submap
    >> -i
-dt deadtimes
    -> writes submap with filled deadtimes
    >> -ipw
-a analyse
    -> run analysis
    >> -ip
-c clustering
    -> run clustering analysis
    >> ip
    
-e export
    -> export pixel data as ascii/numpy blob
    >> -ip
-f input file
    -> assign infile

-o output dir
    -> assign outdir

"""





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
