
import os
import numpy as np
from scipy import ndimage

import xfmkit.bufferops as bufferops
import xfmkit.dtops as dtops
import xfmkit.imgops as imgops
import xfmkit.utils as utils
import xfmkit.config as config

#from ._processed import *

import logging
logger = logging.getLogger(__name__)

#CLASSES
class Xfmap:
    """
    Object wrapping binary file to be read
        holds: params read directly from file
        loads: byte stream from file, holds pointer
        methods to parse pixel header and body, manage memory via chunks
            bufferops.py module contains subsidiary code to parse binary
    """
    def __init__(self, config, fi, fo, WRITE_MODIFIED: bool, CHUNK_SIZE: int, MULTILOAD: bool):

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
        buffer = bufferops.MapBuffer(self.infile, self.chunksize, MULTILOAD)

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
        self.dimensions = ( self.yres, self.xres )

        #config constants
        self.PXHEADERLEN=config['PXHEADERLEN'] 
        self.BYTESPERCHAN=config['BYTESPERCHAN'] 

        self.detarray = bufferops.getdetectors(buffer, self.datastart, self.PXHEADERLEN)
        self.ndet = max(self.detarray)+1

        self.indexlist=np.empty((self.npx, self.ndet),dtype=np.uint64)

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
    def __init__(self, config, xfmap, npx, detarray, INDEX_ONLY: bool):

        #copied variables
        self.source=xfmap
        self.energy=xfmap.energy
        self.dimensions = xfmap.dimensions

        #derived variables
        self.npx = npx
        self.nrows = xfmap.dimensions[0]
        self.nchan=config['NCHAN']

        #assign number of detectors
        self.detarray = detarray
        self.ndet=max(self.detarray)+1

        #initialise pixel value arrays
        self.parsed=False
        self.pxlen=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.xidx=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.yidx=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.det=np.zeros((npx,self.ndet),dtype=np.uint16)
        self.dt=np.zeros((npx,self.ndet),dtype=np.float32)

        #initalise derived arrays
        #flat
        self.flattened=np.zeros((npx),dtype=np.uint32) 
        self.flatsum=np.zeros((npx),dtype=np.uint32) 
        #per-detector
        self.sum=np.zeros((npx,self.ndet),dtype=np.uint32)  
        self.dtmod=np.zeros((npx,self.ndet),dtype=np.float32)  

        #initalise analysis containers
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
            self.data=np.zeros((1024,self.nchan),dtype=np.uint16)
            self.data=None

        self.parsing = INDEX_ONLY


    def receiveheader(self, pxidx, pxlen, xcoord, ycoord, det, dt):
        self.pxlen[pxidx,det]=pxlen
        self.xidx[pxidx,det]=xcoord
        self.yidx[pxidx,det]=ycoord
        self.det[pxidx,det]=det
        self.dt[pxidx,det]=dt
        
        return self

    def truncate_y(self, npx, nrows):

        #find the end of the row
        _current_row = nrows - 1
        _current_pixel = npx - 1
        _x_width = self.dimensions[1]
        
        if not ((_current_row) == (_current_pixel // _x_width)):
            raise ValueError("mismatch between current row and expected row from pixel, dimensions")

        _row_end_index = _current_row * _x_width + _x_width - 1

        new_npx = _row_end_index + 1 

        if not ((new_npx) <= (self.npx)):
            raise ValueError("pixelseries attempting to truncate beyond original number of pixels")

        #do the truncation
        self.npx = new_npx
        self.nrows = nrows
        self.dimensions = ( nrows, self.dimensions[1] )

        self.pxlen=self.pxlen[:new_npx]
        self.xidx=self.xidx[:new_npx]
        self.yidx=self.yidx[:new_npx]
        self.det=self.det[:new_npx]
        self.dt=self.dt[:new_npx]

        #derived arrays
        #flat
        self.flattened=self.flattened[:new_npx]
        self.flatsum=self.flatsum[:new_npx]
        #per-detector
        self.sum=self.sum[:new_npx]
        self.dtmod=self.dtmod[:new_npx]

        #analysis outputs
        self.rvals=self.rvals[:new_npx]
        self.gvals=self.gvals[:new_npx]
        self.bvals=self.bvals[:new_npx]
        self.totalcounts=self.totalcounts[:new_npx]


    def get_dtmod(self, config, xfmap, target_dt: float):
            """
            calculate derived arrays from values extracted from map
            """
            #modify_dt used as both flag and value
            if target_dt < 0:
                #dt = predicted
                self.dtmod = dtops.predict_dt(self, xfmap)
            elif target_dt > 100:
                #dt = unchanged
                self.dtmod = self.dt

            elif target_dt >= 0 and target_dt <= 100:
                #dt = assigned value 0-100
                self.dtmod = np.full((self.dt.shape), np.float32(target_dt), dtype=np.float32) 
            else:
                raise ValueError(f"unexpected value for target_dt, {target_dt}") 

            return self

    def get_derived(self):
        """
        calculate derived arrays from values extracted from map
        """
        self.flattened = np.sum(self.data, axis=1, dtype=np.uint32)
        self.sum = np.sum(self.data, axis=2, dtype=np.uint32)
        self.flatsum = np.sum(self.sum, axis=1, dtype=np.uint32)

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
        write the pixel header statistics
        """
        if config['SAVEFMT_READABLE']:
            for i in self.detarray:
                np.savetxt(os.path.join(dir, "pxstats_pxlen.txt"), self.pxlen, fmt='%i', delimiter=",")
                np.savetxt(os.path.join(dir, "pxstats_xidx.txt"), self.xidx, fmt='%i', delimiter=",")
                np.savetxt(os.path.join(dir, "pxstats_yidx.txt"), self.yidx, fmt='%i', delimiter=",")
                np.savetxt(os.path.join(dir, "pxstats_detector.txt"), self.det, fmt='%i', delimiter=",")
                np.savetxt(os.path.join(dir, "pxstats_dt.txt"), self.dt, fmt='%f', delimiter=",")    
                
                #include derived stats if data was fully parsed
                if self.parsing:
                    np.savetxt(os.path.join(dir, "pxstats_sum.txt"), self.sum, fmt='%d', delimiter=",")  
                    np.savetxt(os.path.join(dir, "pxstats_dtmod.txt"), self.dtmod, fmt='%d', delimiter=",")     
        else:
            np.save(os.path.join(dir, "pxstats_pxlen"), self.pxlen)            
            np.save(os.path.join(dir, "pxstats_xidx"), self.xidx)    
            np.save(os.path.join(dir, "pxstats_yidx"), self.yidx)     
            np.save(os.path.join(dir, "pxstats_det"), self.det)     
            np.save(os.path.join(dir, "pxstats_dt"), self.dt)

            if self.parsing:
                np.save(os.path.join(dir, "pxstats_sum"), self.sum)  
                np.save(os.path.join(dir, "pxstats_dtmod"), self.dtmod)    



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