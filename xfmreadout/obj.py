import struct 
import os
import numpy as np
import json
import copy

import xfmreadout.utils as utils
import xfmreadout.colour as colour
import xfmreadout.fitting as fitting
import xfmreadout.byteops as byteops
import xfmreadout.parser as parser


class MapDone(Exception): pass

#CLASSES
class Xfmap:
    def __init__(self, config, fi, fo):

        #assign input file object for reading
        try:
            self.infile = open(fi, mode='rb') # rb = read binary
            if config['WRITESUBMAP']:
                self.outfile = open(fo, mode='wb')   #wb = write binary
        except FileNotFoundError:
            print("FATAL: incorrect filepath/files not found")

        #get total size of file to parse
        self.fullsize = os.path.getsize(fi)
        self.chunksize = int(config['chunksize'])*int(config['MBCONV'])

        #generate initial bytestream
        #self.stream = self.infile.read()         
        self.stream = self.infile.read(self.chunksize)   
        self.streamlen=len(self.stream)

        #pointers
        self.idx=0      #byte pointer
        self.pxidx=0    #pixel pointer
        self.rowidx=0   #row pointer
        self.pxstart=0  #pointer for start of pixel

        self.fullidx = self.idx
        self.chunkidx = self.idx

        #read the JSON header and move pointer to start of first px record
        self.idx, self.headerdict = parser.readfileheader(self)
        
        #try to assign values from header
        try:
            self.xres = self.headerdict['File Header']['Xres']           #map size x
            self.yres = self.headerdict['File Header']['Yres']           #map size y
            self.xdim = self.headerdict['File Header']['Width (mm)']     #map dimension x
            self.ydim = self.headerdict['File Header']['Height (mm)']    #map dimension y
            self.nchannels = int(self.headerdict['File Header']['Chan']) #no. channels
            self.gain = float(self.headerdict['File Header']['Gain (eV)']/1000) #gain in kV
        except:
            raise ValueError("FATAL: failure reading values from header")
        
        #initialise arrays
        self.chan = np.arange(0,self.nchannels)     #channel series
        self.energy = self.chan*self.gain           #energy series
        self.xarray = np.arange(0, self.xdim, self.xdim/self.xres )   #position series x  
        self.yarray = np.arange(0, self.ydim, self.ydim/self.yres )   #position series y
            #NB: real positions likely better represented by centres of pixels eg. 0+(xdim/xres), xdim-(xdim/xres) 
            #       need to ask IXRF how this is handled by Iridium

        #derived vars
        self.numpx = self.xres*self.yres        #expected number of pixels

        #init struct for reading pixel headers
        self.headstruct=struct.Struct("<ccI3Hf")
        self.PXHEADERLEN=config['PXHEADERLEN'] 
        self.pxlen=self.PXHEADERLEN+config['NCHAN']*4   #dummy value for pxlen

        self.detarray=self.getdetectors(config)

    def getdetectors(self, config):
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
        j=self.idx

        while True:
            #pull stream and extract pixel header
            headstream, j = parser.getstream(self,j,self.PXHEADERLEN)
            pxlen, xidx, yidx, det, dt = parser.readpxheader(headstream, config, self.PXHEADERLEN, self)
            #assign detector
            detarray[i]=int(det)
            #if det=0 for pixel other than 0th, increment and break
            if (i > 0) and (det == 0):
                break
            #otherwise pull next stream to move index and continue
            else:
                readlength=pxlen-self.PXHEADERLEN
                locstream, j = parser.getstream(self,j,readlength)
                i+=1

        return detarray[:i]
    

    def parse(self, config, pxseries):

        print(f"pixels expected: {self.numpx}")
        print("---------------------------")

        if config['WRITESUBMAP']:
            parser.writefileheader(config, self)

        #https://stackoverflow.com/questions/16073396/breaking-while-loop-with-function
        try:
            while True:
                
                headstream, self.idx = parser.getstream(self,self.idx,self.PXHEADERLEN)

                pxlen, xidx, yidx, det, dt = parser.readpxheader(headstream, config, self.PXHEADERLEN, self)

                readlength=pxlen-self.PXHEADERLEN

                pxseries = pxseries.receiveheader(self.pxidx, pxlen, xidx, yidx, det, dt)
            
                locstream, self.idx = parser.getstream(self,self.idx,readlength)

                if config['WRITESUBMAP'] and utils.pxinsubmap(config, xidx, yidx):
                        parser.writepxheader(config, self, pxseries, det)
                        parser.writepxrecord(locstream, readlength, self)

                if config['PARSEMAP']:
                    chan, counts = parser.readpxdata(locstream, config, readlength)

                    #fill gaps in spectrum 
                    #   (ie. assign all zero-count chans = 0)
                    chan, counts = utils.gapfill(chan,counts, config['NCHAN'])

                    #warn if recieved channel list is different length to chan array
                    if len(chan) != len(self.chan):
                        print("WARNING: unexpected length of channel list")

                    #store pixel sum
                    pxseries.sum[det,self.pxidx]=np.sum(counts)

                    #assign counts into data array
                    pxseries.data[det,self.pxidx,:]=counts

                self.fullidx=self.chunkidx+self.idx

                #if on last detector for this pixel, increment counters and check end
                if det == max(self.detarray):
                    #stop when pixel index greater than expected no. pixels
                    if (self.pxidx >= (self.numpx-1)):
                        print(f"\nENDING AT: Row {self.rowidx}/{self.yres} at pixel {self.pxidx}")
                        raise MapDone

                    #print pixel index at end of every row
                    if self.pxidx % self.xres == (self.xres-1): 
                        print(f"\rRow {self.rowidx}/{self.yres-1} at pixel {self.pxidx}, byte {self.fullidx} ({100*self.fullidx/self.fullsize:.1f} %)", end='')
                        self.rowidx+=1

                    self.pxidx+=1    #next pixel
        except MapDone:
            #store no. pixels and rows read successfully
            pxseries.npx=self.pxidx+1
            pxseries.nrows=self.rowidx+1 

            return pxseries

    def nextchunk(self):
        #NB: chunkdx likely broken after refactor
        self.chunkidx = self.chunkidx + self.idx

        self.stream = self.infile.read(self.chunksize)

        if len(self.stream) != self.streamlen:
            print("\n NOTE: final chunk")

        self.streamlen=len(self.stream)
        self.idx=0

        if not self.stream:
            print(f"\n WARNING: Attempting to load chunk beyond EOF - dimensions in header may be incorrect.")
            raise MapDone

    def closefiles(self, config):
        self.infile.close()
        if config['WRITESUBMAP']:
            self.outfile.close()


class PixelSeries:
    def __init__(self, config, xfmap):

        #assign number of detectors
        ndet=max(xfmap.detarray)+1

        #initialise pixel value arrays
        self.pxlen=np.zeros((ndet,xfmap.numpx),dtype=np.uint16)
        self.xidx=np.zeros((ndet,xfmap.numpx),dtype=np.uint16)
        self.yidx=np.zeros((ndet,xfmap.numpx),dtype=np.uint16)
        self.det=np.zeros((ndet,xfmap.numpx),dtype=np.uint16)
        self.sum=np.zeros((ndet,xfmap.numpx),dtype=np.uint32)        
        self.dt=np.zeros((ndet,xfmap.numpx),dtype=np.float32)

        #create colour-associated attrs even if not doing colours
        self.rvals=np.zeros(xfmap.numpx)
        self.gvals=np.zeros(xfmap.numpx)
        self.bvals=np.zeros(xfmap.numpx)
        self.totalcounts=np.zeros(xfmap.numpx)

        #initialise whole data containers (WARNING: large)
        if config['PARSEMAP']: 
            self.data=np.zeros((ndet,xfmap.numpx,config['NCHAN']),dtype=np.uint16)
#            if config['DOBG']: self.corrected=np.zeros((xfmap.numpx,config['NCHAN']),dtype=np.uint16)
        else:
        #create a small dummy array just in case
            self.data=np.zeros((1024,config['NCHAN']),dtype=np.uint16)

        self.npx=0
        self.nrows=0

    def receiveheader(self, pxidx, pxlen, xcoord, ycoord, det, dt):
        self.pxlen[det,pxidx]=pxlen
        self.xidx[det,pxidx]=xcoord
        self.yidx[det,pxidx]=ycoord
        self.det[det,pxidx]=det
        self.dt[det,pxidx]=dt
        
        return self

    def flatten(self, data, detarray):
        """
        sum all detectors into single data array
        NB: i think this creates another dataset in memory while running
        """
        flattened = data[0]
        if len(detarray) > 1:
            for i in detarray[1:]:
                flattened+=data[i]
        
        return flattened

    def exportheader(self, config, odir):
        parser.exportheader(config, self, odir)

    def exportseries(self, config, odir):
        parser.exportseries(config, self, odir)

    def readseries(self, config, odir):
        self = parser.readseries(config, self, odir)
        return self