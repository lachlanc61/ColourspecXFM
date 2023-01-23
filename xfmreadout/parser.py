import struct 
import os
import numpy as np
import json
import copy

import xfmreadout.byteops as byteops


def readfileheader(xfmap):
    print(
        "---------------------------\n"
        f"PARSING HEADER\n"
        "---------------------------"
    )

    streamlen=len(xfmap.stream)
    print(f"filesize: {streamlen} (bytes)")
    #if beginning of file
    #   read header length from first bytes as <uint16
    if xfmap.idx == 0:
        headerlen=byteops.binunpack(xfmap,"<H")
    else:
        raise ValueError("FATAL: attempting to read file header from nonzero index")

    #check for header
    #   pixels start with "DP" (=20550 as <uint16)
    #   if we find this immediately, header is zero length - cannot proceed
    #provided header is present
    #   read params from header
    if headerlen == 20550:  #(="DP" as <uint16)
        raise ValueError("FATAL: file header missing, cannot read map params")
    #also fail if headerlength is below arbitrary value
    elif headerlen <= 500:
        raise ValueError("FATAL: file header too small, check input")
    #else proceed
    else:
        """
        if header present, read as json
        https://stackoverflow.com/questions/40059654/python-convert-a-bytes-array-into-json-format
        """
        #pull slice of byte stream corresponding to header
        #   bytes[0-2]= headerlen
        #   doesn't include trailing '\n' '}', so +2
        headerraw=xfmap.stream[2:headerlen+2]

        #read it as utf8
        headerstream = headerraw.decode('utf-8')
        
        #load into dictionary via json builtin
        headerdict = json.loads(headerstream)

    #print map params
    print(f"header length: {headerlen} (bytes)")

    #set pointer index to length of header + 2 bytes
    xfmap.idx = headerlen+2

    return xfmap.idx, headerdict

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

def getstream(xfmap, idx, length):

    #if we have enough remaining in the chunk, proceed (CHECK not sure if > or >=)
    if not idx+length > xfmap.streamlen:    
        locstream=xfmap.stream[idx:idx+length]
        idx=idx+length
    else:   #if step would exceed chunk
        #read the remainder of the chunk
        locstream=xfmap.stream[idx:xfmap.streamlen]
        #store the length read
        gotlen=xfmap.streamlen-idx  

        #load next chyunk and update xfmap idx
        xfmap.nextchunk() 

        #load the remainder of the pixel
        locstream+=xfmap.stream[0:length-gotlen]

        idx = length - gotlen

    if len(locstream) < length:
        raise ValueError("FATAL: received stream shorter than expected, unexpected EOF")

    return locstream, idx

def readpxheader(headstream, config, readlength, xfmap):
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
    headstream=headstream[:readlength]

    #unpack the header
    #   faster to unpack into temp variables vs directly into pbject attrs. not sure why atm
    pxflag0, pxflag1, pxlen, xidx, yidx, det, dt = xfmap.headstruct.unpack(headstream)

    #   check for pixel start flag "DP":
    pxflag=pxflag0+pxflag1
    if not (pxflag == b'DP'):
        raise ValueError(f"ERROR: pixel flag 'DP' expected but not found for pixel {xfmap.pxidx}")

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

def readseries(config, pxseries, odir):
    """
    read data from a pre-saved datfile
        does not currently return as much information as the full parse
    """
    print("loading from file", config['outfile'])
    pxseries.data = np.loadtxt(os.path.join(odir, config['outfile']), dtype=np.uint16)
    pxseries.pxlen=np.loadtxt(os.path.join(odir, "pxlen.txt"), dtype=np.uint16)
    pxseries.xidx=np.loadtxt(os.path.join(odir, "xidx.txt"), dtype=np.uint16)
    pxseries.yidx=np.loadtxt(os.path.join(odir, "yidx.txt"), dtype=np.uint16)
    pxseries.det=np.loadtxt(os.path.join(odir, "detector.txt"), dtype=np.uint16)
    pxseries.dt=np.loadtxt(os.path.join(odir, "dt.txt"), dtype=np.uint16)
    print("loaded successfully", config['outfile']) 

    return pxseries

def exportseries(config, pxseries, odir):
    print("saving spectrum-by-pixel to file")
    np.savetxt(os.path.join(odir,  config['outfile'] + ".dat"), pxseries.data, fmt='%i')    

def exportheader(config, pxseries, odir):
    np.savetxt(os.path.join(odir, "pxlen.txt"), pxseries.pxlen, fmt='%i')
    np.savetxt(os.path.join(odir, "xidx.txt"), pxseries.xidx, fmt='%i')
    np.savetxt(os.path.join(odir, "yidx.txt"), pxseries.yidx, fmt='%i')
    np.savetxt(os.path.join(odir, "detector.txt"), pxseries.det, fmt='%i')
    np.savetxt(os.path.join(odir, "dt.txt"), pxseries.dt, fmt='%i')

def writepxheader(config, xfmap, pxseries):
    pxflag=config['PXFLAG']
    pxflag0=pxflag[0].encode(config['CHARENCODE'])
    pxflag1=pxflag[1].encode(config['CHARENCODE'])
    pxlen=pxseries.pxlen[xfmap.pxidx]
    xcoord=pxseries.xidx[xfmap.pxidx]
    ycoord=pxseries.yidx[xfmap.pxidx]
    det=pxseries.det[xfmap.pxidx]
    dt=pxseries.dt[xfmap.pxidx]

    #write the header with x/y coords adjusted
    outstream=xfmap.headstruct.pack(pxflag0,pxflag1, pxlen, xcoord-config['submap_x'][0], \
                                    ycoord-config['submap_y'][0], det, dt)
    xfmap.outfile.write(outstream)

        # write the channel data as-is
        

def writepxrecord(locstream, readlength, xfmap):
    xfmap.outfile.write(locstream[:readlength])


def endrow(xfmap):
    xfmap.fullidx=xfmap.chunkidx+xfmap.idx
    print(f"Row {xfmap.rowidx}/{xfmap.yres-1} at pixel {xfmap.pxidx}, byte {xfmap.fullidx} ({100*xfmap.fullidx/xfmap.fullsize:.1f} %)", end='\r')
    xfmap.rowidx+=1    