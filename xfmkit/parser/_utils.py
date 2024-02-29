
import time
import numpy as np

import xfmkit.bufferops as bufferops
import xfmkit.utils as utils
import xfmkit.structures as structures

class MapDone(Exception): pass

def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    """
    Cleanup operations at end of each pixel

        prints status at end of each row

        raises MapDone at expected end of map
    """
    row=pixelseries.yidx[pxidx,0]+1

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


def readspectrum(buffer,det,absidx,pxlength,pxheaderlen,bytesperchan,nchannels):
#def processpixel(buffer,det,pxidx,indexlist,pxheaderlen,bytesperchan,nchannels):

    relidx=int(absidx-buffer.fidx)

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
        ___, counts = bufferops.readpxdata(stream, len(stream), bytesperchan, nchannels)
    finally:
        return buffer, counts
    #    pixelseries.data[pxidx,det,:]=counts

    ___ = endpx(pxidx, absidx, buffer, xfmap, pixelseries)    