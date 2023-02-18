import numpy as np

import xfmreadout.bufferops as bufferops
import xfmreadout.utils as utils



class MapDone(Exception): pass

def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    """
    Cleanup operations at end of each pixel

        prints status at end of each row

        raises MapDone at expected end of map
    """
    row=pixelseries.yidx[0,pxidx]+1

    #print pixel index at end of every row
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
        pixelseries.dtflat = np.sum(pixelseries.dt, axis=0)/pixelseries.ndet

        buffer.wait()
        xfmap.resetfile()
        return pixelseries, indexlist


def parse(xfmap, pixelseries, indexlist, multiproc):
    """
    read in the map data after indexing

    NB, getstream is returning stream incl. pixel header > failing
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
                
                absidx=indexlist[det,pxidx]
                relidx=int(absidx-buffer.fidx)
                pxlength=pixelseries.pxlen[det,pxidx]

                if relidx < 0:
                        raise ValueError(f"pixel start {absidx} not in current buffer beginning {buffer.fidx}") 

                try:
                    #if read exceeds buffer, get next buffer via getstream
                    if relidx+pxlength > buffer.len:

                        #FUTURE: join here to wait for processes
                        #should be able to parallelize pullstream + readpx+gapfill + assign
                        
                        #if break is in header, cycle buffer via getstream
                        if relidx+pxheaderlen > buffer.len:     #not sure about > vs >=
                            ___, ___, buffer=bufferops.getstream(buffer, relidx, pxheaderlen)
                        
                        #get next stream
                        stream, ___, buffer=bufferops.getstream(buffer, relidx+pxheaderlen, pxlength-pxheaderlen)
                    #otherwise read it directly
                    else:

                        stream = buffer.data[relidx+pxheaderlen:relidx+pxlength]

                    #continue
                    chan, counts = bufferops.readpxdata(stream, len(stream), bytesperchan)
                    chan, counts = utils.gapfill(chan,counts, xfmap.nchannels)  #<-should probably go in readpxdata
                except ValueError:  #not sure I need this, really just a debug log
                    print(f"{det}, {pxidx}")
                    exit()
                finally:
                    pixelseries.data[det,pxidx,:]=counts
            ___ = endpx(pxidx, absidx, buffer, xfmap, pixelseries)
    except MapDone:
        if not pixelseries.npx == pxidx+1:
            raise ValueError(f"Index mistmatch ({pixelseries.npx}) vs ({pxidx})")
        if not pixelseries.nrows == pixelseries.yidx[0,pxidx]+1:
            raise ValueError(f"Index mistmatch ({pixelseries.nrows}) vs ({pixelseries.yidx[0,pxidx]+1})")
        buffer.wait()
        xfmap.resetfile()
        
        return pixelseries


def writemap(config, xfmap, pixelseries, xcoords, ycoords, dtfill, multiproc):
    """
    Write a map or submap
        Updates headers and pixel headers
            !Does not change pixel data!

        Crops to coordinates
        Fills/predicts/corrects deadtimes if needed
    """

    #write file header
    bufferops.writefileheader(xfmap, xcoords, ycoords)

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

            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[det,pxidx], pixelseries.yidx[det,pxidx]):            
                bufferops.writepxheader(config, xfmap, pixelseries, det, pxidx, xcoords, ycoords, dtfill)
           
            stream, idx, buffer = bufferops.getstream(buffer, idx, pixelseries.pxlen[det,pxidx]-pxheaderlen)
            
            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[det,pxidx], pixelseries.yidx[det,pxidx]):            
                bufferops.writepxrecord(xfmap, stream, pixelseries.pxlen[det,pxidx]-pxheaderlen)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        buffer.wait()
        xfmap.resetfile()
        return 