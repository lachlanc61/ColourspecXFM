import numpy as np

import xfmreadout.bufferops as bufferops
import xfmreadout.utils as utils



class MapDone(Exception): pass

def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    #print pixel index at end of every row
    row=pixelseries.yidx[0,pxidx]+1

    if pxidx % xfmap.xres == (xfmap.xres-1): 
        print(f"\rRow {row}/{xfmap.yres} at pixel {pxidx}, byte {int(buffer.fidx+idx)} ({100*(idx)/xfmap.fullsize:.1f} %)", end='')
        pass
    #stop when pixel index greater than expected no. pixels
    if (pxidx >= (xfmap.npx-1)):
        print(f"\nEND OF MAP: row {row}/{xfmap.yres}, pixel {pxidx}")
        raise MapDone

    pxidx+=1

    return pxidx


def initparse(xfmap, multiproc):
    """
    unused for now
    """
    xfmap.resetfile()
    buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiproc)
    idx = xfmap.datastart
    pxheaderlen = xfmap.PXHEADERLEN    

    return buffer, idx, pxheaderlen

def indexmap(xfmap, pixelseries, multiproc):
    """
    parse the pixel headers
    - get pixel statistics
    index the file
    """
    print("--------------")
    print("INDEXING")
    try:
        indexlist=np.zeros((pixelseries.ndet,xfmap.npx),dtype=np.uint64)

        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiproc)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN    

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)
            
            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(headstream)

            indexlist[det, pxidx] = buffer.fidx+idx-pxheaderlen

            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)
            
            #use getstream to step to the next pixel and handle end-of-buffer events
            __, idx, buffer = bufferops.getstream(buffer, idx, pxlen-pxheaderlen)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        pixelseries.npx=pxidx+1
        pixelseries.nrows=pixelseries.yidx[0,pxidx]+1 
        buffer.wait()
        xfmap.resetfile()
        return pixelseries, indexlist


def parse(xfmap, pixelseries, indexlist, multiproc):
    """
    read in the map data after indexing

    NB. no longer getting sum from parser, do this in postana using pixelseries.data
    """
    print("--------------")
    print("PARSING PIXEL DATA")
    try:
        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiproc)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN
        bytesperchan = xfmap.BYTESPERCHAN

        for pxidx in range(pixelseries.npx):
            for det in range(pixelseries.ndet):

                if not det == pixelseries.det[det,pxidx]:
                    raise ValueError(f"Detector mistmatch at (det,pixel) = ({det},{pxidx})")
                
                start=indexlist[det,pxidx]
                pxlength=pixelseries.pxlen[det,pxidx]

                try:
                #should be able to parallelize pullstream + readpx+gapfill + assign
                    if start+pxlength <= buffer.fidx + buffer.len:
                        stream = bufferops.pullstream(buffer, start, pxlength, pxheaderlen)
                    else:
                        #join
                        stream, ___, buffer=bufferops.getstream(buffer, int(start-buffer.fidx), pxlength)

                    chan, counts = bufferops.readpxdata(stream, len(stream), bytesperchan)
                    chan, counts = utils.gapfill(chan,counts, xfmap.nchannels)  #<-should probably go in readpxdata
                except ValueError:
                    print(f"{det}, {pxidx}")
                    exit()
                finally:
                    pixelseries.data[det,pxidx,:]=counts
            ___ = endpx(pxidx, start, buffer, xfmap, pixelseries)
    except MapDone:
        pixelseries = pixelseries.get_derived()
        
        if not pixelseries.npx == pxidx+1:
            raise ValueError(f"Index mistmatch ({pixelseries.npx}) vs ({pxidx})")
        if not pixelseries.nrows == pixelseries.yidx[0,pxidx]+1:
            raise ValueError(f"Index mistmatch ({pixelseries.nrows}) vs ({pixelseries.yidx[0,pxidx]+1})")
        buffer.wait()
        xfmap.resetfile()
        
        return pixelseries


def writemap(config, xfmap, pixelseries, xcoords, ycoords, multiproc):
    """
    Write a map or submap
        Updates headers and pixel headers
            !Does not change pixel data!

        Crops to coordinates
        Fills/predicts/corrects deadtimes if needed
    """

    xstart=xcoords[0]
    xend=xcoords[1]
    ystart=ycoords[0]
    yend=ycoords[1]

    #write file header
    bufferops.writefileheader(config, xfmap)

    print("--------------")
    print("EXPORTING MAP")
    try:
        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiproc)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)

            ___, xidx, yidx, det, ___ = bufferops.readpxheader(headstream)

            if not [ xidx, yidx, det ] == [ pixelseries.xidx[det,pxidx], pixelseries.yidx[det,pxidx], pixelseries.det[det,pxidx] ]:
                raise ValueError(f"values read from pixel header do not match result from indexing")

            if utils.pxinsubmap(config, pixelseries.xidx[det,pxidx], pixelseries.yidx[det,pxidx]):            
                bufferops.writepxheader(config, xfmap, pixelseries, det, pxidx)
           
            stream, idx, buffer = bufferops.getstream(buffer, idx, pixelseries.pxlen[det,pxidx]-pxheaderlen)
            
            if utils.pxinsubmap(config, pixelseries.xidx[det,pxidx], pixelseries.yidx[det,pxidx]):            
                bufferops.writepxrecord(xfmap, stream, pixelseries.pxlen[det,pxidx]-pxheaderlen)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        buffer.wait()
        xfmap.resetfile()
        return 