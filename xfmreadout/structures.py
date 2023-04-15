import os
import numpy as np

import xfmreadout.bufferops as bufferops
import xfmreadout.dtops as dtops

#CLASSES
class Xfmap:
    """
    Object wrapping binary file to be read
        holds: params read directly from file
        loads: byte stream from file, holds pointer
        methods to parse pixel header and body, manage memory via chunks
            bufferops.py module contains subsidiary code to parse binary
    """
    def __init__(self, config, fi, fo, WRITE_MODIFIED: bool, CHUNK_SIZE: int, MULTIPROC: bool):

        #assign input file object for reading
        try:
            self.infile = open(fi, mode='rb') # rb = read binary
            if WRITE_MODIFIED:
                self.writing = True
                self.outfile = open(fo, mode='wb')   #wb = write binary
            else:
                self.writing = False
        except FileNotFoundError:
            print("FATAL: incorrect filepath/files not found")

        #get total size of file to parse
        self.fullsize = os.path.getsize(fi)
        self.chunksize = CHUNK_SIZE

        self.fidx=self.infile.tell()

        if self.fidx != 0:
            raise ValueError(f"File pointer at {self.fidx} - Expected 0 (start of file)")

        #read the beginning of the file into buffer
        buffer = bufferops.MapBuffer(self.infile, self.chunksize, MULTIPROC)

        #read the JSON header and store position of first pixel
        self.headerdict, self.datastart, buffer = bufferops.readjsonheader(buffer, 0)
        
        #try to assign values from header
        try:
            self.xres = int(self.headerdict['File Header']['Xres'])           #map size x
            self.yres = int(self.headerdict['File Header']['Yres'])           #map size y
            self.xdim = float(self.headerdict['File Header']['Width (mm)'])     #map dimension x
            self.ydim = float(self.headerdict['File Header']['Height (mm)'])    #map dimension y
            self.nchannels = int(self.headerdict['File Header']['Chan']) #no. channels
            self.gain = float(self.headerdict['File Header']['Gain (eV)']/1000) #gain in kV
            self.deadtime = float(self.headerdict['File Header']['Deadtime (%)'])
            self.dwell = float(self.headerdict['File Header']['Dwell (mS)'])   #dwell in ms
            self.timeconst = float(config['time_constant']) #pulled from config, ideally should be in header
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
        self.npx = self.xres*self.yres        #expected number of pixels

        #config constants
        self.PXHEADERLEN=config['PXHEADERLEN'] 
        self.BYTESPERCHAN=config['BYTESPERCHAN'] 

        self.detarray = bufferops.getdetectors(buffer, self.datastart, self.PXHEADERLEN)
        self.maxdet = max(self.detarray)

        buffer.wait()
        self.resetfile()
        return

    def resetfile(self):
        self.infile.seek(0)

    def closefiles(self):
        self.infile.close()
        if self.writing:
            self.outfile.close()



class PixelSeries:
    def __init__(self, config, xfmap, npx, detarray, INDEX_ONLY):

        self.source=xfmap
        self.energy=xfmap.energy

        #assign number of detectors
        self.detarray = detarray
        self.npx = npx
        self.ndet=max(self.detarray)+1
        self.nchan=config['NCHAN']

        #initialise pixel value arrays
        self.pxlen=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.xidx=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.yidx=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.det=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.dt=np.zeros((npx,self.ndet),dtype=np.float32)

        #initalise derived arrays
        #flat
        self.flattened=np.zeros((npx),dtype=np.uint32) 
        self.flatsum=np.zeros((npx),dtype=np.uint32) 
        self.dtflat=np.zeros((npx),dtype=np.float32)  
        #per-detector
        self.sum=np.zeros((npx,self.ndet),dtype=np.uint32)  
        self.dtpred=np.zeros((npx,self.ndet),dtype=np.float32)  

        #create analysis outputs
        self.rvals=np.zeros(npx)
        self.gvals=np.zeros(npx)
        self.bvals=np.zeros(npx)
        self.totalcounts=np.zeros(npx)

        #dummy arrays for outputs
        self.categories=np.zeros(10)
        self.classavg=np.zeros(10)
        self.rgbarray=np.zeros(10)      
        self.corrected=np.zeros(10)

        #initialise whole data containers (WARNING: large)
        if not INDEX_ONLY:
            self.data=np.zeros((npx,self.ndet,self.nchan),dtype=np.uint16)
#            if config['DOBG']: self.corrected=np.zeros((xfmap.npx,config['NCHAN']),dtype=np.uint16)
        else:
        #create a small dummy array just in case
            self.data=np.zeros((1024,self.nchan),dtype=np.uint16)

        self.parsing = INDEX_ONLY

        self.npx=0
        self.nrows=0

    def receiveheader(self, pxidx, pxlen, xcoord, ycoord, det, dt):
        self.pxlen[pxidx,det]=pxlen
        self.xidx[pxidx,det]=xcoord
        self.yidx[pxidx,det]=ycoord
        self.det[pxidx,det]=det
        self.dt[pxidx,det]=dt
        
        return self

    def get_derived(self, config, xfmap):
        """
        calculate derived arrays from values extracted from map
        """
        self.flattened = np.sum(self.data, axis=1, dtype=np.uint32)
        self.sum = np.sum(self.data, axis=2, dtype=np.uint32)
        self.flatsum = np.sum(self.sum, axis=1, dtype=np.uint32)

        self.dtpred = dtops.predict_dt(config, self, xfmap)

        return self


    def flatten_REMOVE(self, data, detarray):
        """
        sum all detectors into single data array
        NB: i think this creates another dataset in memory while running
        PRETTY SURE NOT USED - confirm
        """
        flattened = data[0]
        if len(detarray) > 1:
            for i in detarray[1:]:
                flattened+=data[i]
        
        return flattened

    def exportpxstats(self, config, dir):
        """
        write the pixel header statistics to csv
        """
        np.savetxt(os.path.join(dir, "pxstats_pxlen.txt"), self.pxlen, fmt='%i', delimiter=",")
        np.savetxt(os.path.join(dir, "pxstats_xidx.txt"), self.xidx, fmt='%i', delimiter=",")
        np.savetxt(os.path.join(dir, "pxstats_yidx.txt"), self.yidx, fmt='%i', delimiter=",")
        np.savetxt(os.path.join(dir, "pxstats_detector.txt"), self.det, fmt='%i', delimiter=",")
        np.savetxt(os.path.join(dir, "pxstats_dt.txt"), self.dt, fmt='%f', delimiter=",")    
        
        #include derived stats if data was fully parsed
        if self.parsing:
            np.savetxt(os.path.join(dir, "pxstats_sum.txt"), self.sum, fmt='%d', delimiter=",")  
            np.savetxt(os.path.join(dir, "pxstats_dtpred.txt"), self.dtpred, fmt='%d', delimiter=",")    
            np.savetxt(os.path.join(dir, "pxstats_dtflat.txt"), self.dtflat, fmt='%d', delimiter=",")  


    def exportpxdata(self, config, dir):
        """
        writes the spectrum-by-pixel data to csv
        """
        if config['SAVEFMT_READABLE']:
            for i in self.detarray:
                np.savetxt(os.path.join(dir,  config['export_filename'] + f"{i}.txt"), self.data[i], fmt='%i')
        else:
            np.save(os.path.join(dir,  config['export_filename']), self.data)


    def importpxdata(self, config, dir):
        """
        read data from csv
            does not currently return as much information as the full parse

        NB: currently broken after refactor
        """
        print("loading from file", config['export_filename'])
        self.data = np.loadtxt(os.path.join(dir, config['outfile']), dtype=np.uint16, delimiter=",")
        self.pxlen=np.loadtxt(os.path.join(dir, "pxstats_pxlen.txt"), dtype=np.uint16, delimiter=",")
        self.xidx=np.loadtxt(os.path.join(dir, "pxstats_xidx.txt"), dtype=np.uint16, delimiter=",")
        self.yidx=np.loadtxt(os.path.join(dir, "pxstats_yidx.txt"), dtype=np.uint16, delimiter=",")
        self.det=np.loadtxt(os.path.join(dir, "pxstats_detector.txt"), dtype=np.uint16, delimiter=",")
        self.dt=np.loadtxt(os.path.join(dir, "pxstats_dt.txt"), dtype=np.float32, delimiter=",")
        
        print("loaded successfully", config['export_filename']) 

        return self