import numpy as np

import xfmreadout.structures as structures
import xfmreadout.bufferops as bufferops


class MapDone(Exception): pass

def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    #print pixel index at end of every row
    row=pixelseries.yidx[0,pxidx]+1

    if pxidx % xfmap.xres == (xfmap.xres-1): 
        print(f"\rRow {row}/{xfmap.yres} at pixel {pxidx}, byte {buffer.fidx+idx} ({100*(buffer.fidx+idx)/xfmap.fullsize:.1f} %)", end='')
        pass
    #stop when pixel index greater than expected no. pixels
    if (pxidx >= (xfmap.npx-1)):
        print(f"\nENDING AT: Row {row}/{xfmap.yres} at pixel {pxidx}")
        raise MapDone

    pxidx+=1

    return pxidx


def indexmap(xfmap, pixelseries):
    print("BEGIN INDEXING")
    try:
        indexlist=np.zeros((pixelseries.ndet,xfmap.npx),dtype=np.uint64)

        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)
            
            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(headstream)

            indexlist[det, pxidx] = idx-pxheaderlen

            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)
            
            #use getstream to step to the next pixel and handle end-of-buffer events
            __, idx, buffer = bufferops.getstream(buffer, idx, pxlen-pxheaderlen)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx, buffer, xfmap, pixelseries)

    except MapDone:
        pixelseries.npx=pxidx+1
        pixelseries.nrows=pixelseries.yidx[0,pxidx]+1 
        xfmap.resetfile()
        return pixelseries, indexlist
