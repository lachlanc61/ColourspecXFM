import numpy as np

import xfmreadout.bufferops as bufferops
import xfmreadout.utils as utils
import xfmreadout.parallel as parallel

import parsercore

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
        indexlist=np.empty((xfmap.npx, xfmap.ndet),dtype=np.uint64) #must match xfmap.indexlist declaration

        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiproc)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN    

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)
            
            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(headstream)

            indexlist[pxidx, det] = buffer.fidx+idx-pxheaderlen

            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)
            
            #use getstream to step to the next pixel and handle end-of-buffer events
            __, idx, buffer = bufferops.getstream(buffer, idx, pxlen-pxheaderlen)

            if det == xfmap.ndet-1:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        pixelseries.npx=pxidx+1
        pixelseries.nrows=pixelseries.yidx[pxidx,0]+1 
        pixelseries.dtflat = np.sum(pixelseries.dt, axis=1)/pixelseries.ndet

        buffer.wait()
        xfmap.resetfile()
        return pixelseries, indexlist


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



def parse(xfmap, pixelseries, multiproc):
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
        indexlist = xfmap.indexlist
        pxlen=pixelseries.pxlen

#        indices_ravel = np.ravel(indexlist, order='F')
#        pxlens_ravel = np.ravel(pixelseries.pxlen, order='F')

        if not multiproc:
            for pxidx in range(pixelseries.npx):
                for det in range(pixelseries.ndet):

                    if not det == pixelseries.det[pxidx,det]:
                        raise ValueError(f"Detector mistmatch at (pixel,det) = ({pxidx},{det})")

                    absidx=indexlist[pxidx,det]
                    pxlength=pixelseries.pxlen[pxidx,det]

                    #read spectrum and update buffer if needed
                    buffer, pixelseries.data[pxidx,det,:] = readspectrum(buffer,det,absidx,pxlength,pxheaderlen,bytesperchan,nchannels)

                ___ = endpx(pxidx, absidx, buffer, xfmap, pixelseries)

        #PARALLELIZED
        else: 
            buffer_start_px = 0

            while buffer_start_px <= xfmap.fullsize:

                #index_final = indexlist + pixelseries.pxlen
                
                #store original buffer start and end
                buffer_start=buffer.fidx
                buffer_end=buffer.fidx+buffer.len

                #find pixel containing end of buffer
                buffer_break_px = np.searchsorted(indexlist[:,0], buffer_end) - 1 
                                        # -1 because np.ssorted gives first row of next buffer
                buffer_end_px = buffer_break_px - 1
                                        # -1 again to get last unbroken pixel
                                        
                #get stream indexes and lengths corresponding to buffer
                stream_indexes=indexlist[buffer_start_px:buffer_break_px,:]-buffer.fidx
                stream_pxlen=pxlen[buffer_start_px:buffer_break_px,:]

                print(f"\nReading buffer from pixels {buffer_start_px} to {buffer_break_px}")             
                
                #extract the data
                stream_data = parsercore.readstream(stream_indexes, stream_pxlen, buffer.data, len(buffer.data))

                #copy it to pixelseries
                pixelseries.data[buffer_start_px:buffer_break_px,:,:] = stream_data

                #read in final pixel manually
                #   update buffer via readspectrum
                for det in range(pixelseries.ndet):
                    absidx=indexlist[buffer_break_px,det]
                    pxlength=pixelseries.pxlen[buffer_break_px,det]

                    #read spectrum and update buffer if needed
                    buffer, pixelseries.data[buffer_break_px,det,:] = readspectrum(buffer,det,absidx,pxlength,pxheaderlen,bytesperchan,nchannels)

                #check that buffer has changed
                if buffer_start == buffer.fidx:
                    #check if we are at end of file
                    if buffer_end == xfmap.fullsize:
                        print(f"\nEND OF MAP: pixel {buffer_break_px}")
                        raise MapDone
                    else:
                        raise ValueError(f"Buffer position unchanged at end of pixel")
                else:
                    #increment beginning of next buffer
                    buffer_start_px = buffer_break_px+1

    except MapDone:

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

            if not [ xidx, yidx, det ] == [ pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det], pixelseries.det[pxidx,det] ]:
                raise ValueError(f"values read from pixel header do not match result from indexing")

            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxheader(config, xfmap, pixelseries, det, pxidx, xcoords, ycoords, dtfill)
           
            stream, idx, buffer = bufferops.getstream(buffer, idx, pixelseries.pxlen[pxidx,det]-pxheaderlen)
            
            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxrecord(xfmap, stream, pixelseries.pxlen[pxidx,det]-pxheaderlen)

            if det == xfmap.ndet-1:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except MapDone:
        buffer.wait()
        xfmap.resetfile()
        return 