import struct 
import sys
import numpy as np
import json
import copy

import multiprocessing as mp

import xfmkit.byteops as byteops
import xfmkit.parser as parser
import xfmkit.utils as utils

import logging
logger = logging.getLogger(__name__)

#assign an identifier to the local namespace
#   used to create persistent preloaded buffer
this = sys.modules[__name__]

pxheadstruct = struct.Struct("<ccI3Hf")

def worker(infile, chunksize, pipe_child):
    """
    Worker for multiprocess
        loads new buffer object in a subprocess to cache it 
        then sends data and file index via pipe to MapBuffer.retrieve()
            will hold open until data is received
            to take data as dummy and close, use MapBuffer.wait()
    
    NB: reads from infile concurrently, make sure completed before main
        reinitialises buffer or moves file head
    """
    nextbuffer=MapBuffer(infile, chunksize, True)
    pipe_child.send(nextbuffer.data)
    pipe_child.send(nextbuffer.fidx)
    pipe_child.close()
    #should wait here until received

class MapBuffer:
    """
    Object holding current chunk of file for processing
    """
    def __init__(self, infile, chunksize: int, multiproc: bool):
        self.infile=infile
        self.fidx = self.infile.tell()

        if (self.fidx == 0):
            cache_flag = True
        else:
            cache_flag = False

        self.chunksize=chunksize
        self.multiproc=multiproc
        self.len = 0
        try:
            self.data = self.infile.read(self.chunksize) 
            self.len = len(self.data)
            #time.sleep(0.1)            
        except:
            raise EOFError(f"No data to load from {self.infile}")

        self.check()

        if self.multiproc and cache_flag:
            self.cache()

        return

    def cache(self):
        """
        Spawn new multiprocess to pre-load next chunk
        Create a pipe to transfer newly loaded chunk
        """
        #print('Caching...')
        self.pipe_parent, self.pipe_child = mp.Pipe()   
        self.process = mp.Process(target=worker, args=(self.infile, self.chunksize, self.pipe_child))
        self.process.start()

    def retrieve(self):
        """
        Receives data and file index from next chunk preloaded by Multiprocess

            Waits for process to complete, close pipes
            Assign new data to current buffer
            Begin caching next chunk
        """
        if self.multiproc:
            nextdata=self.pipe_parent.recv()
            nextfidx=self.pipe_parent.recv()

            self.process.join()    
            self.pipe_parent.close
            self.pipe_child.close

            self.data = nextdata
            self.fidx = nextfidx
            self.len = len(nextdata)

            self.cache()
        else:
            try:
                self.fidx = self.infile.tell()
                self.data = self.infile.read(self.chunksize) 
                self.len = len(self.data)
            except:
                raise EOFError(f"No data to load from {self.infile}")    

        self.check()

        return self

    def check(self):
        """
        simple sanity checks/readouts on self
            run after new chunk is loaded
        """
        if self.len < self.chunksize and not self.len == 0:
            #print("\n NOTE: final chunk")
            pass
        
        if self.data == "":
            print(f"\n WARNING: Attempting to load chunk beyond EOF - dimensions in header may be incorrect.")
            raise parser.MapDone

        return

    def wait(self):
        """
        wait for running cache to complete
        """
        if self.multiproc:
            #need to receive sends, otherwise will block indefinitely
            ___=self.pipe_parent.recv()
            ___=self.pipe_parent.recv()
            self.process.join()
        else:
            pass

def getstream(buffer, idx: int, length: int):
    """
    get the next stream, loading new buffer if current read would exceed buffer length
    """
    #if we have enough remaining in the chunk, proceed (CHECK not sure if > or >=)
    if idx+length <= buffer.len:    
        stream=buffer.data[idx:idx+length]
        idx=idx+length
    else:   #if step would exceed chunk
        #read the remainder of the chunk
        stream=buffer.data[idx:buffer.len]
        #store the length read
        gotlen=buffer.len-idx  

        buffer = buffer.retrieve()

        #load the remainder of the pixel
        stream+=buffer.data[0:length-gotlen]

        idx = length - gotlen

        a=1

    if len(stream) < length:
        if buffer.len == 0:
            print(f"\n WARNING: Mismatch between EOF and expected pixel count - map dimensions may be incorrect in file header.")
            raise parser.MapDone
        else:
            raise ValueError("FATAL: unexpected stream size before end of buffer")
        
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


def readpxdata(stream, readlength, bytesperchan: int, nchannels: int):
    """
    read in data from a single pixel

    takes: 
        stream containing pixel data
        the length to read in (starting at 0)
        the expected no. bytes per datachannel

    returns: 
        sparse output channels and counts
    """

    #initialise channel index and result arrays
    chan=np.zeros(int((readlength)/bytesperchan), dtype=int)
    counts=np.zeros(int((readlength)/bytesperchan), dtype=int)

    #generate struct object for reading
    fmt= "<%dH" % ((readlength) // (bytesperchan/2))
    chanstruct=struct.Struct(fmt)

    if len(stream) != readlength:
        raise ValueError("stream size does not match struct")

    #use it to read the channel data
    chandata=chanstruct.unpack(stream[:readlength])

    #take even indexes for channels
    chan=chandata[::2]
    
    #take odd indexes for counts
    counts=chandata[1::2]

    chan, counts = utils.gapfill(list(chan), list(counts), nchannels)

    #return as lists
    return chan, counts


def writefileheader(xfmap, xcoords, ycoords):
    """
    writes the main header for the file

        adjusts for cropped coordinates
    """

    #modify width and height in header and re-print

    xstart = xcoords[0]
    if xcoords[1] <= xfmap.xres:
        xend = xcoords[1]
    else:
        xend = xfmap.xres

    newxres=xend-xstart
    #if new res larger than original, set to original
    if newxres > xfmap.xres:
        print("WARNING: derived X size larger than X size read from data")
        newxres = xfmap.xres
    newxdim=newxres*(xfmap.headerdict["File Header"]["Width (mm)"]/xfmap.headerdict["File Header"]["Xres"])


    ystart = ycoords[0]
    if ycoords[1] <= xfmap.yres:
        yend = ycoords[1]
    else:
        yend = xfmap.yres

    newyres=yend-ystart

    #if new res larger than original, set to original
    if newyres > xfmap.yres:
        print("WARNING: derived Y size larger than Y size read from data")        
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


def writepxheader(config, xfmap, pxseries, det: int, pxidx: int, xcoords, ycoords, modify_dt:float):
    """
    write the header for a single pixel
    takes:
        - the config,  XfMap and PixelSeries objects
        - the detector and pixel index to write into header
        - x and y coordinate arrays
        - whether we are calculating new deadtimes

    """
    if modify_dt == -1:
        dt=pxseries.dt[pxidx,det]
    else:
        dt=pxseries.dtmod[pxidx,det]

    pxflag=config['PXFLAG']
    pxflag0=pxflag[0].encode(config['CHARENCODE'])
    pxflag1=pxflag[1].encode(config['CHARENCODE'])

    pxlen=pxseries.pxlen[pxidx,det]
    xidx=pxseries.xidx[pxidx,det]
    yidx=pxseries.yidx[pxidx,det]
    
    if pxseries.det[pxidx,det] != det:
        raise ValueError("WARNING: Detector value mismatch")

    #write the header with x/y coords adjusted
    outstream=pxheadstruct.pack(pxflag0,pxflag1, pxlen, xidx-xcoords[0], \
                                    yidx-ycoords[0], det, dt)
    xfmap.outfile.write(outstream)
        

def writepxrecord(xfmap, stream, length):
    """
    write the pixel data directly from stream
    """
    xfmap.outfile.write(stream[:length])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      