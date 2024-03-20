import time
import numpy as np
import logging

import parsercore

import xfmkit.bufferops as bufferops
import xfmkit.utils as utils
import xfmkit.structures as structures

from ._utils import *

logger = logging.getLogger(__name__)


DEBUG = False


def indexmap(xfmap, pixelseries, multiload):
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
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiload)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN    

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)
            
            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(headstream)

            indexlist[pxidx, det] = buffer.fidx+idx-pxheaderlen

            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)

            #use getstream to step to the next pixel and handle end-of-buffer events
            __spectrum_stream, idx, buffer = bufferops.getstream(buffer, idx, pxlen-pxheaderlen)

            if det == xfmap.ndet-1:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)          

    except MapComplete:
        """
        handle expected end of map (raised during endpx())
        """

        #assign final sizes
        npx=pxidx+1     #pixel index still on final pixel
        nrows=pixelseries.yidx+1 

        if not pixelseries.npx == xfmap.npx:
            print("WARNING: pixelseries and map object have different pixel sizes at clean completion")

        pixelseries.npx = npx
        pixelseries.nrows = nrows
        pixelseries.dimensions = ( nrows, pixelseries.dimensions[1] )        

        xfmap.indexlist = indexlist
        xfmap.npx_found = xfmap.npx

        buffer.wait()
        xfmap.resetfile()
        return pixelseries, xfmap

    except MapEarlyStop:
        """
        handle early end of map (raised from bufferops eg. getstream)
        """        
        npx = pxidx #pxidx progressed to next (nonexistent) pixel
        nrows = yidx+1  #yidx still on last even if end of row

        print("Resizing dataset to match size of indexed map")

        if not (npx == xfmap.npx and nrows == xfmap.yres ):
            pixelseries.truncate_y(npx, nrows)
            xfmap.indexlist = indexlist[:npx]
            xfmap.npx_found = npx
        else:
            print("WARNING: map sizes match despite early stop")

        buffer.wait()
        xfmap.resetfile()
        return pixelseries, xfmap


def parse(xfmap, pixelseries, multiload):
    """
    read in the map data after indexing

    NB, getstream is returning stream incl. pixel header > failing
    """
    print("--------------")
    print("PARSING PIXEL DATA")
    try:
        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiload)
        idx = xfmap.datastart

        pxheaderlen = xfmap.PXHEADERLEN
        bytesperchan = xfmap.BYTESPERCHAN
        nchannels = xfmap.nchannels

        #CHECK might not work
        indexlist = xfmap.indexlist
        pxlen=pixelseries.pxlen

#        indices_ravel = np.ravel(indexlist, order='F')
#        pxlens_ravel = np.ravel(pixelseries.pxlen, order='F')

        python_only = False
        if python_only:
            print(f"\nParsing via Python")            
            for pxidx in range(pixelseries.npx):
                for det in range(pixelseries.ndet):

                    if not det == pixelseries.det[pxidx,det]:
                        raise ValueError(f"Detector mistmatch at (pixel,det) = ({pxidx},{det})")

                    absidx=indexlist[pxidx,det]
                    pxlength=pixelseries.pxlen[pxidx,det]

                    #read spectrum and update buffer if needed
                    buffer, pixelseries.data[pxidx,det,:] = readspectrum(buffer,det,absidx,pxlength,pxheaderlen,bytesperchan,nchannels)

                ___ = endpx(pxidx, absidx, buffer, xfmap, pixelseries)

        #using C++
        else:         
            print("Reading .GeoPIXE file via C++")
            buffer_start_px = 0
            print(f"\nParsing via C++")
            while buffer_start_px <= xfmap.fullsize:

                #index_final = indexlist + pixelseries.pxlen
                
                #store original buffer start and end
                buffer_start=buffer.fidx
                buffer_end=buffer.fidx+buffer.len

                #find pixel containing end of buffer
                buffer_break_px = np.searchsorted(indexlist[:,0], buffer_end) - 1 
                                        # -1 because np.ssorted gives first row of next buffer
                
                #get last unbroken pixel
                if indexlist[buffer_break_px,-1] + pxlen[buffer_break_px,-1] == buffer_end:
                    #if buffer ends perfectly at end of pixel, last good pixel is break px
                    buffer_last_px =  buffer_break_px
                elif indexlist[buffer_break_px,-1] + pxlen[buffer_break_px,-1] < buffer_end:
                    print(f"WARNING: unexplained data at end of buffer - last pixel byte: {indexlist[buffer_break_px,-1] + pxlen[buffer_break_px,-1]}, buffer end byte: {buffer_end}")
                    buffer_last_px = buffer_break_px
                else:
                    buffer_last_px = buffer_break_px - 1

                #get stream indexes and lengths corresponding to buffer up to last unbroken pixel
                stream_indexes=indexlist[buffer_start_px:buffer_last_px+1,:]-buffer.fidx
                stream_pxlen=pxlen[buffer_start_px:buffer_last_px+1,:]

                print(f"\nReading buffer, pixels {buffer_start_px} to {buffer_last_px}")             
                
                if DEBUG == True:
                    print(f"\nStart of data to read: {buffer.data[:100]}")
                    print(f"\nFirst pixel location: {indexlist[buffer_start_px, 0]}")   
                    print(f"\nFirst pixel byte: {buffer.data[int(indexlist[buffer_start_px, 0]):(int(indexlist[buffer_start_px, 0])+8)]}") 
                    print(f"\nTypes for parsercore: {type(stream_indexes), stream_indexes.dtype}, {type(stream_pxlen), stream_pxlen.dtype}, {type(buffer.data)}, {type(len(buffer.data))},")

                 #extract the data
                #---------------------
                parsed_stream = parsercore.readstream(stream_indexes, stream_pxlen, buffer.data, len(buffer.data))
                #---------------------

                #copy it to pixelseries
                pixelseries.data[buffer_start_px:buffer_last_px+1,:,:] = parsed_stream

                #read in final pixel manually to update buffer if needed
                #   (will have already been read if stream has pixel perfect end)
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
                        print(f"WARNING: unexpected early stop - last pixel byte: {indexlist[buffer_break_px,-1] + pxlen[buffer_break_px,-1]}, map end byte: {xfmap.fullsize}")
                        raise MapEarlyStop(f"Buffer position unchanged at end of pixel")
                else:
                    #increment beginning of next buffer
                    buffer_start_px = buffer_break_px+1

    except ( MapDone, MapComplete, MapEarlyStop ) :
        pixelseries.parsed = True
        buffer.wait()
        xfmap.resetfile()
        return pixelseries
    

def writemap(config, xfmap, pixelseries, xcoords, ycoords, modify_dt, multiload):
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
    print("WRITING NEW .GeoPIXE FILE")
    try:
        xfmap.resetfile()
        buffer = bufferops.MapBuffer(xfmap.infile, xfmap.chunksize, multiload)
        idx = xfmap.datastart
        pxheaderlen = xfmap.PXHEADERLEN

        pxidx=0
        while True:

            headstream, idx, buffer = bufferops.getstream(buffer, idx, pxheaderlen)

            ___, xidx, yidx, det, ___ = bufferops.readpxheader(headstream)

            if not [ xidx, yidx, det ] == [ pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det], pixelseries.det[pxidx,det] ]:
                raise ValueError(f"values read from pixel header do not match result from indexing")

            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxheader(config, xfmap, pixelseries, det, pxidx, xcoords, ycoords, modify_dt)
           
            stream, idx, buffer = bufferops.getstream(buffer, idx, pixelseries.pxlen[pxidx,det]-pxheaderlen)
            
            if utils.pxinsubmap(xcoords, ycoords, pixelseries.xidx[pxidx,det], pixelseries.yidx[pxidx,det]):            
                bufferops.writepxrecord(xfmap, stream, pixelseries.pxlen[pxidx,det]-pxheaderlen)

            if det == xfmap.ndet-1:
                pxidx = endpx(pxidx, idx+buffer.fidx, buffer, xfmap, pixelseries)

    except ( MapDone, MapComplete, MapEarlyStop ) :
        buffer.wait()
        xfmap.resetfile()
        return pixelseries        