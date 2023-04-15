import numpy as np

import xfmreadout.bufferops as bufferops
import xfmreadout.utils as utils
import xfmreadout.parallel as parallel

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
        indexlist=np.zeros((xfmap.npx, pixelseries.ndet),dtype=np.uint64)

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
        pixelseries.nrows=pixelseries.yidx[pxidx,0]+1 
        pixelseries.dtflat = np.sum(pixelseries.dt, axis=1)/pixelseries.ndet

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
        nchannels = xfmap.nchannels

        #CHECK might not work
        indices_ravel = np.ravel(indexlist, order='F')
        pxlens_ravel = np.ravel(pixelseries.pxlen, order='F')

        if not multiproc:
            for pxidx in range(pixelseries.npx):
                for det in range(pixelseries.ndet):

                    if not det == pixelseries.det[pxidx,det]:
                        raise ValueError(f"Detector mistmatch at (pixel,det) = ({pxidx},{det})")
                    
                    absidx=indexlist[pxidx,det]
                    relidx=int(absidx-buffer.fidx)
                    pxlength=pixelseries.pxlen[pxidx,det]

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
                        chan, counts = bufferops.readpxdata(stream, len(stream), bytesperchan, nchannels)

                    except ValueError:  #not sure I need this, really just a debug log
                        print(f"{pxidx}, {det}")
                        exit()
                    finally:
                        pixelseries.data[pxidx,det,:]=counts
                ___ = endpx(pxidx, absidx, buffer, xfmap, pixelseries)

            #PARALLELIZED
        else: 

            index_final = indexlist + pixelseries.pxlen
            
            buffer_start=buffer.fidx
            buffer_end=buffer.fidx+buffer.len

            start_idx_flat=np.searchsorted(indices_ravel, buffer_start)
            start_idx=divmod(start_idx_flat, indexlist.shape[0])


            end_idx_flat=np.searchsorted(indices_ravel, buffer_end)-1   #-1 to get last idx fully within buffer
            end_idx=divmod(end_idx_flat, indexlist.shape[0])

#            start_coords=unravel(indexlist, buffer_start, True)
#            end_coords=unravel(indexlist, buffer_end, False)

            indices_sliced = indices_ravel[start_idx_flat:end_idx_flat] 
            pxlens_sliced = pxlens_ravel[start_idx_flat:end_idx_flat] 

            worker_array = parallel.worker(buffer.data, indices_sliced, pxlens_sliced, pxheaderlen, bytesperchan, nchannels)


    except MapDone:
        if not pixelseries.npx == pxidx+1:
            raise ValueError(f"Index mistmatch ({pixelseries.npx}) vs ({pxidx})")
        if not pixelseries.nrows == pixelseries.yidx[pxidx,0]+1:
            raise ValueError(f"Index mistmatch ({pixelseries.nrows}) vs ({pixelseries.yidx[pxidx,0]+1})")
        buffer.wait()
        xfmap.resetfile()
        
        return pixelseries


def unravel(indexlist, max_value: int, is_start: bool):
    """
    UNUSED
    
    extracts 2D index of indexlist where pixel is fully within buffer

    https://stackoverflow.com/questions/22565023/numpy-searchsorted-with-2d-array
        ravel order F interleaves arrays eg. ( 1 3 5 ),(2 4 6) => (1 2 3 4 5 6)
        searchsorted finds index where buffer_end would be inserted
        divmod shape0 converts back to original coords
                        
    """
    NDET=2

    ravel=np.ravel(indexlist, order='F')
    
    if is_start == True:
        shift=0     #0 to start
    else:
        shift=-1    #-1 to get last idx fully within buffer

    idx_flat=np.searchsorted(ravel, max_value)
    idx_flat+=shift    
    coords=divmod(idx_flat, indexlist.shape[0])

    return coords[::-1]


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

            if not [ xidx, yidx, det ] == [ pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det], pixelseries.det[pxidx,det] ]:
                raise ValueError(f"values read from pixel header do not match result from indexing")

            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxheader(config, xfmap, pixelseries, det, pxidx, xcoords, ycoords, dtfill)
           
            stream, idx, buffer = bufferops.getstream(buffer, idx, pixelseries.pxlen[pxidx,det]-pxheaderlen)
            
            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxrecord(xfmap, stream, pixelseries.pxlen[pxidx,det]-pxheaderlen)

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        buffer.wait()
        xfmap.resetfile()
        return 