import struct 
import os
import numpy as np
import json
import copy

import xfmreadout.byteops2 as byteops
import xfmreadout.dtops as dtops
import xfmreadout.obj2 as obj


class MapDone(Exception): pass

pxheadstruct=struct.Struct("<ccI3Hf")

def getbuffer(infile, chunksize):
    buffer = obj.MapBuffer(infile, chunksize)

    if buffer.len < buffer.chunksize:
        print("\n NOTE: final chunk")
    
    if not buffer.data:
        print(f"\n WARNING: Attempting to load chunk beyond EOF - dimensions in header may be incorrect.")
        raise MapDone

    return buffer      

def getstream(buffer, idx, length):
    #if we have enough remaining in the chunk, proceed (CHECK not sure if > or >=)
    if not idx+length > buffer.len:    
        stream=buffer.data[idx:idx+length]
        idx=idx+length
    else:   #if step would exceed chunk
        #read the remainder of the chunk
        stream=buffer.data[idx:buffer.len]
        #store the length read
        gotlen=buffer.len-idx  

        buffer = getbuffer(buffer.infile, buffer.chunksize) 

        #load the remainder of the pixel
        stream+=buffer.data[0:length-gotlen]

        idx = length - gotlen

    if len(stream) < length:
        raise ValueError("FATAL: received stream shorter than expected")

    return stream, idx, buffer


def getdetectors(buffer, idx, pxheaderlen):
    """
    Find the detector config for the map

    Pre-parses self.stream and jumps through pixel headers, 
        saving detector values, until detectors begin to repeat

    NB: assumes:
        - detectors increase sequentially and are uniform throughout file
            eg. 0 1 2 3 repeating pixel-by-pixel
        - first pixel is representative of config for whole map
    """
    #initialise array and counters
    detarray=np.zeros(20).astype(int)
    i=0

    while True:
        #pull stream and extract pixel header
        headstream, idx, buffer = getstream(buffer,idx,pxheaderlen)
        pxlen, xidx, yidx, det, dt = readpxheader(headstream)
        #assign detector
        detarray[i]=int(det)
        #if det=0 for pixel other than 0th, increment and break
        if (i > 0) and (det == 0):
            break
        #otherwise pull next stream to move index and continue
        else:
            idx+=pxlen-pxheaderlen
            i+=1

    return detarray[:i]

def readjsonheader(buffer, idx):

        if idx != 0:
            print(f"WARNING: header being read from byte {idx}. Expected beginning of file.")

        headerlen=byteops.binunpack(buffer.data, idx, "<H")

        if headerlen == 20550:  #(="DP" as <uint16)
            raise ValueError("FATAL: file header missing, cannot read map params")
        elif headerlen <= 500:
            raise ValueError("FATAL: file header too small, check input")
        else:
            #proceed

            #pull slice of byte stream corresponding to header
            #   bytes[0-2]= headerlen
            #   doesn't include trailing '\n' '}', so +2
            headerraw, idx, buffer = getstream(buffer, 2, headerlen)

            #read it as utf8
            headerstream = headerraw.decode('utf-8')
            
            #load into dictionary via json builtin
            headerdict = json.loads(headerstream)

        #print map params
        print(f"header length: {headerlen} (bytes)")

        #set pointer index to length of header + 2 bytes -> position of first pixel record
        idx = headerlen+2

        return headerdict, idx, buffer


def readpxheader(headstream):
    """"
    Pixel Record
    Note: not name/value pairs for file size reasons. The pixel record header is the only record type name/value pair, for easier processing. We are keeping separate records for separate detectors, since the deadtime information will also be per detector per pixel.
        1.	Record type pair  "DP", Length of pixel data record in bytes ( 4 byte int)
        2.	X                          Horizontal pixel index (2 byte int)
        3.	Y                          Vertical pixel index (2 byte int)
        4.	Detector               Data in this record is for this detector (2 byte int)
        5.	Deadtime             Deadtime for this pixel (4 byte float)
        6.	Data (for each channel with data up to maximum channel index)
            a.	Channel     Channel index (0- Max Chan) (2 byte int)
            b.	Count         Event counts in channel (2 byte int)

    #   concise format:
    #   DP  len     X       Y       det     dt  DATA
    #   2c  4i     2i       2i      2i      4f
    """
    """
    Read binary with struct
    https://stackoverflow.com/questions/8710456/reading-a-binary-file-with-python
    Read binary as chunks
    https://stackoverflow.com/questions/71978290/python-how-to-read-binary-file-by-chunks-and-specify-the-beginning-offset
    """

    #unpack the header
    #   faster to unpack into temp variables vs directly into object attrs. not sure why atm
    pxflag0, pxflag1, pxlen, xidx, yidx, det, dt = pxheadstruct.unpack(headstream)

    #   check for pixel start flag "DP":
    pxflag=pxflag0+pxflag1
    if not (pxflag == b'DP'):
        raise ValueError(f"ERROR: pixel flag 'DP' expected but not found")

    return pxlen, xidx, yidx, det, dt


def readpxdata(locstream, config, readlength):

    #initialise channel index and result arrays
    chan=np.zeros(int((readlength)/config['BYTESPERCHAN']), dtype=int)
    counts=np.zeros(int((readlength)/config['BYTESPERCHAN']), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #create struct object for reading
    fmt= "<%dH" % ((readlength) // 2)
    chanstruct=struct.Struct(fmt)

    #read the channel data
    chandata=chanstruct.unpack(locstream[:readlength])
    #take even indexes for channels
    chan=chandata[::2]
    #take odd indexes for counts
    counts=chandata[1::2]

    #return as lists
    return(list(chan), list(counts))


def endpx(pxidx, idx, buffer, xfmap, pixelseries):
    #print pixel index at end of every row
    row=pixelseries.yidx[0,pxidx]+1

    if pxidx % xfmap.xres == (xfmap.xres-1): 
        print(f"\rRow {row}/{xfmap.yres} at pixel {pxidx}, byte {buffer.fidx+idx} ({100*(buffer.fidx+idx)/xfmap.fullsize:.1f} %)", end='')
        pass
    #stop when pixel index greater than expected no. pixels
    if (pxidx >= (xfmap.numpx-1)):
        print(f"\nENDING AT: Row {row}/{xfmap.yres} at pixel {pxidx}")
        raise MapDone

    pxidx+=1

    return pxidx


def indexmap(xfmap, pixelseries):
    try:

        xfmap.resetfile()
        buffer = getbuffer(xfmap.infile, xfmap.chunksize)
        idx = xfmap.datastart
        pxidx=0
        while True:

            headstream, idx, buffer = getstream(buffer, idx, xfmap.PXHEADERLEN)
            
            pxlen, xidx, yidx, det, dt = readpxheader(headstream)
            
            pixelseries = pixelseries.receiveheader(pxidx, pxlen, xidx, yidx, det, dt)
            
            idx+=pxlen-xfmap.PXHEADERLEN

            if det == xfmap.maxdet:
                pxidx = endpx(pxidx, idx, buffer, xfmap, pixelseries)

    except MapDone:
        pixelseries.npx=pxidx+1
        pixelseries.nrows=pixelseries.yidx[0,pxidx]+1 
        xfmap.resetfile()
        return pixelseries















def writefileheader(config, xfmap):
    #modify width and height in header and re-print

    newxres=config['submap_x'][1]-config['submap_x'][0]
    #if new res larger than original, set to original
    if newxres > xfmap.xres:
        newxres = xfmap.xres
    newxdim=newxres*(xfmap.headerdict["File Header"]["Width (mm)"]/xfmap.headerdict["File Header"]["Xres"])

    newyres=config['submap_y'][1]-config['submap_y'][0]
    #if new res larger than original, set to original
    if newyres > xfmap.yres:
        newyres = xfmap.yres
    newydim=newyres*(xfmap.headerdict["File Header"]["Height (mm)"]/xfmap.headerdict["File Header"]["Yres"])

    #create a duplicate via deepcopy
    #   need deepcopy because nested lists - normal copy would point to original data still
    newheaderdict = copy.deepcopy(xfmap.headerdict)
    newheaderdict["File Header"]["Xres"]=newxres
    newheaderdict["File Header"]["Width (mm)"]=newxdim
    newheaderdict["File Header"]["Yres"]=newyres
    newheaderdict["File Header"]["Height (mm)"]=newydim

    #create a printable version  
    headerdump = json.dumps(newheaderdict, indent='\t', sort_keys=False)
    #create a byte-encoded version 
    headerencode = headerdump.encode('utf-8')

    #write the new header length
    xfmap.outfile.write(struct.pack("<H",len(headerencode)))
    #write the new header
    xfmap.outfile.write(headerencode)

    #NB: PROBLEM HERE ----------------
    # The default JSON has a duplicate entry.
    # "Detector" appears twice beacuse there are two dets
    # first is overwritten during json.loads
    #   .: only one in dict to write to second file
    #   think we can ignore this, the info is not used, but header is different when rewritten


def writepxheader(config, xfmap, pxseries, det):

    dt=pxseries.dt[det,xfmap.pxidx]

    #assign or predict deadtime values if not present
    if config['FILL_DT'] and pxseries.ndet == 1:
        if dt > 0:
            raise ValueError("WARNING: Found nonzero deadtime while overwriting")
        #fill with fixed value
        dt=float(config['assign_dt'])

    elif config['PREDICT_DT'] and pxseries.ndet == 1:
        if dt > 0:
            raise ValueError("WARNING: Found nonzero deadtime while overwriting")
        #predict from pixel sum
        dt=dtops.predictdt(config, pxseries.sum[det,xfmap.pxidx], xfmap.dwell, xfmap.timeconst)
    else:
        #leave unchanged
        dt=dt

    pxflag=config['PXFLAG']
    pxflag0=pxflag[0].encode(config['CHARENCODE'])
    pxflag1=pxflag[1].encode(config['CHARENCODE'])

    pxlen=pxseries.pxlen[det,xfmap.pxidx]
    xcoord=pxseries.xidx[det,xfmap.pxidx]
    ycoord=pxseries.yidx[det,xfmap.pxidx]
    usedet=pxseries.det[det,xfmap.pxidx]
    
    if usedet != det:
        raise ValueError("WARNING: Detector value mismatch")

    #write the header with x/y coords adjusted
    outstream=xfmap.headstruct.pack(pxflag0,pxflag1, pxlen, xcoord-config['submap_x'][0], \
                                    ycoord-config['submap_y'][0], det, dt)
    xfmap.outfile.write(outstream)
        

def writepxrecord(locstream, readlength, xfmap):
    xfmap.outfile.write(locstream[:readlength])


def endrow(xfmap):
    xfmap.fullidx=xfmap.chunkidx+xfmap.idx
    print(f"Row {xfmap.rowidx}/{xfmap.yres-1} at pixel {xfmap.pxidx}, byte {xfmap.fullidx} ({100*xfmap.fullidx/xfmap.fullsize:.1f} %)", end='\r')
    xfmap.rowidx+=1    