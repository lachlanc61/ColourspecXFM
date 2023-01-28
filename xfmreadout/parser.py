import numpy as np

import xfmreadout.obj as obj
import xfmreadout.bufferops as bufferops




def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    #print pixel index at end of every row
    row=pixelseries.yidx[0,pxidx]+1

    if pxidx % xfmap.xres == (xfmap.xres-1): 
        print(f"\rRow {row}/{xfmap.yres} at pixel {pxidx}, byte {buffer.fidx+idx} ({100*(buffer.fidx+idx)/xfmap.fullsize:.1f} %)", end='')
        pass
    #stop when pixel index greater than expected no. pixels
    if (pxidx >= (xfmap.numpx-1)):
        print(f"\nENDING AT: Row {row}/{xfmap.yres} at pixel {pxidx}")
        raise bufferops.MapDone

    pxidx+=1

    return pxidx


def indexmap(xfmap, pixelseries):
    try:

        xfmap.resetfile()
        buffer = bufferops.getbuffer(xfmap.infile, xfmap.chunksize)
        idx = xfmap.datastart
        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, xfmap.PXHEADERLEN)
            
            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(headstream)
            
            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)
            
            #use getstream to step to the next pixel and handle end-of-buffer events
            __, idx, buffer = bufferops.getstream(buffer, idx, pxlen-xfmap.PXHEADERLEN)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx, buffer, xfmap, pixelseries)

    except bufferops.MapDone:
        pixelseries.npx=pxidx+1
        pixelseries.nrows=pixelseries.yidx[0,pxidx]+1 
        xfmap.resetfile()
        return pixelseries
