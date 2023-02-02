import numpy as np



"""
if args
    use args
if not args
    use config
always use protocol




modularise - refactoring config to use flags/kwargs

always index 
    -> save pixel header data

-p parse 
    -> save pixel header data + sums, flatsums
    >> 
-w write
    -> write submap
    >> 
-dt deadtimes
    -> writes submap with filled deadtimes
    >> pw
-a analyse
    -> run analysis
    >> p
-c clustering
    -> run clustering analysis
    >>p

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
