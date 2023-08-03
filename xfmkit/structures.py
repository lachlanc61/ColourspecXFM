import os
import numpy as np
from scipy import ndimage

import xfmkit.bufferops as bufferops
import xfmkit.dtops as dtops
import xfmkit.imgops as imgops
import xfmkit.utils as utils

from math import sqrt
from math import log

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


        #assign number of detectors
        self.detarray = detarray
        self.npx = npx
        self.ndet=max(self.detarray)+1
        self.nchan=config['NCHAN']

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
        self.dtflat=np.zeros((npx),dtype=np.float32)  
        #per-detector
        self.sum=np.zeros((npx,self.ndet),dtype=np.uint32)  
        self.dtmod=np.zeros((npx,self.ndet),dtype=np.float32)  

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

        #self.nrows=0

    def receiveheader(self, pxidx, pxlen, xcoord, ycoord, det, dt):
        self.pxlen[pxidx,det]=pxlen
        self.xidx[pxidx,det]=xcoord
        self.yidx[pxidx,det]=ycoord
        self.det[pxidx,det]=det
        self.dt[pxidx,det]=dt
        
        return self

    def get_dtmod(self, config, xfmap, modify_dt: float):
            """
            calculate derived arrays from values extracted from map
            """
            #modify_dt used as both flag and value
            if modify_dt == -1:
                #dt = unchanged
                self.dtmod = self.dt
            elif modify_dt == 999:
                #dt = predicted
                self.dtmod = dtops.predict_dt(config, self, xfmap)
            elif modify_dt >= 0 and modify_dt <= 100:
                #dt = assigned value 0-100
                self.dtmod = np.full((self.dt.shape), np.float32(modify_dt), dtype=np.float32) 
            else:
                raise ValueError("unexpected value for modify_dt") 

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
                    np.savetxt(os.path.join(dir, "pxstats_dtflat.txt"), self.dtflat, fmt='%d', delimiter=",")  
        else:
            np.save(os.path.join(dir, "pxstats_pxlen"), self.pxlen)            
            np.save(os.path.join(dir, "pxstats_xidx"), self.xidx)    
            np.save(os.path.join(dir, "pxstats_yidx"), self.yidx)     
            np.save(os.path.join(dir, "pxstats_det"), self.det)     
            np.save(os.path.join(dir, "pxstats_dt"), self.dt)

            if self.parsing:
                np.save(os.path.join(dir, "pxstats_sum"), self.sum)  
                np.save(os.path.join(dir, "pxstats_dtmod"), self.dtmod)    
                np.save(os.path.join(dir, "pxstats_dtflat"), self.dtflat) 



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


def data_unroll(maps):
    """
    reshape map (x, y, counts) to data (i, counts)

    returns dataset and dimensions
    """

    if len(maps.shape) == 3:
        data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
        dims=maps.shape[:2]
    elif len(maps.shape) == 2:
        data=maps.reshape(maps.shape[0]*maps.shape[1])
        dims=maps.shape[:2]        
    else:
        raise ValueError(f"unexpected dimensions for {map}")

    return data, dims    


class DataSeries:
    def __init__(self, data: 'np.ndarray', dimensions=None):
        """
        linked pair of 1D dataset and 2D image stack that are views of each other
        """    
            
        self.d, self.dimensions = self.import_by_shape(data, dimensions=dimensions)

        self.shape = self.d.shape
        #TO DO: check C-contiguous and copy to new dataset if not
      
        #assign a 2D view for image-based operations
        self.mapview = self.mapview_from_data(self.d, self.dimensions)

        self.dtype = self.d.dtype

        self.check()

    def check(self):
        """
        basic checks on dataset    
        """

        if not np.issubdtype(self.d.dtype, np.number):
            raise ValueError("data for DataSeries must be numeric")

        if not len(self.d.shape) == 2:
            raise ValueError("invalid data shape")

        if not self.shape == self.d.shape:
            raise ValueError("shape mismatch between dataseries and data")          

        if not len(self.mapview.shape) == 3:
            raise ValueError("invalid maps shape")
        
        if not self.d.shape[1] == self.mapview.shape[2]:
            raise ValueError("mismatch between data and map channels")
        
        if not self.d.shape[0] == self.mapview.shape[0]*self.mapview.shape[1]:
            raise ValueError("mismatch between data and map shapes")
        
        if not self.dimensions == (self.mapview.shape[0], self.mapview.shape[1]):
            raise ValueError("mismatch between specified dimensions and map shape")
        
        if not np.may_share_memory(self.d, self.mapview):
            raise ValueError("dataseries data and mapview have become disconnected")
        
        return True

    def import_by_shape(self, data, dimensions=None):
        """
        ingest an array and extract data and dimensions

        array can be either 3D Y,X,NCHAN or 2D N, NCHAN

        unrolled dimensions must be given for 2D map   
        """

        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("data for DataSeries must be numeric")

        #if 2D data and 2 dimensions given, proceed as N,CHAN map with explicit dims
        if len(data.shape) == 2 and len(dimensions) == 2:
            data_ = data
            dimensions_ = dimensions
        
        #if 3D data given with matching dimensions, proceed as Y,X,CHAN map with explicit dims
        elif len(data.shape) == 3 and dimensions == (data.shape[0], data.shape[1]):
            data_ = data.reshape(data.shape[0]*data.shape[1],-1)
            dimensions_ = dimensions

        #if 3D data given without dimensions, proceed as Y,X,CHAN map and derive dimensions
        elif dimensions == None and len(data.shape) == 3:
            data_, dimensions_ = self.data_from_mapview(data)

        #fail cases:
        elif dimensions == None and not len(data.shape) == 3:  
            raise ValueError("2D dataset provided without explicit map dimensions")          
        else:
            raise ValueError(f"Unexpected shapes for data {data.shape} and dimensions {dimensions}")

        return data_, dimensions_

    def fill_from(self, data: 'np.ndarray'):
        """
        Ingests data into self.d without reallocating memory
        
        Preserves mapview
        
        data must be correct shape
        """
        if self.dtype == data.dtype:
            data_ = data
        else:
            data_ = utils.smartcast(data, self.dtype)

        if len(data_.shape) == 2:
            if self.d.shape == data_.shape:
                self.d[:,:] = data_
            else:
                raise ValueError(f"incompatible shapes {self.d.shape} vs {data_.shape}")
       
        elif len(data_.shape) == 3:
            if self.mapview.shape == data_.shape:
                self.mapview[:,:,:] = data_
            else:
                raise ValueError(f"incompatible shapes {self.mapview.shape} vs {data_.shape}")
        else:
            raise ValueError(f"unexpected dimensionality for ingested array")

        self.check()
    
    def set_to(self, data: 'np.ndarray'):
        """
        reassigns self.d to point to new data

        functionally similar to creating new DataSeries
        checks shapes/dimensions match

        """
        data_ = data

        if len(data_.shape) == 2:
            if self.d.shape == data_.shape:
                self.d = data_
                self.mapview = self.mapview_from_data(data_, self.dimensions)
            else:
                raise ValueError(f"incompatible shapes {self.d.shape} vs {data_.shape}")
 
        elif len(data_.shape) == 3:
            if self.mapview.shape == data_.shape:
                self.mapview = data_
                self.d, dimensions_ = self.data_from_mapview(data_)

                if not dimensions_ == self.dimensions:
                    raise ValueError(f"reassigning with different dimensions {self.dimensions} vs {dimensions_}")
                    
            else:
                raise ValueError(f"incompatible shapes {self.mapview.shape} vs {data_.shape}")
        else:
            raise ValueError(f"unexpected dimensionality for array being assigned")

        self.dtype = self.d.dtype        

        self.check()

    def data_from_mapview(self, mapview):
        """
        reshape mapview into data and return with original dimensions
        """
        d_ = mapview.reshape(mapview.shape[0]*mapview.shape[1],-1)
        dimensions_ = (mapview.shape[0], mapview.shape[1])
        return d_, dimensions_

    def mapview_from_data(self, d, dimensions):
        """
        reshape data into mapview based on dimensions
        """
        mapview_ = d.reshape(dimensions[0], dimensions[1], -1)
        return mapview_

    def crop(self, xrange=(0, 99999), yrange=(0, 99999)):
        """
        crop maps in 2D and adjustcorresponding 1D view
        """
        self.mapview = self.mapview[yrange[0]:yrange[1], xrange[0]:xrange[1], :]
        self.d, self.dimensions = self.data_from_mapview(self.mapview)
        self.shape = self.d.shape

    def zoom(self, zoom_factor, order:int = None):
        """
        scale maps in 2D based on zoom factor
        """
        if order == None:   #if no order given, guess from zoom factor
            if zoom_factor < 1:    
                order = 1   #bicubic for downsampling
            else:
                order = 2   #bilinear for upsampling

        zoom_params = (zoom_factor, zoom_factor, 1) #do not resize on axis 3 (= channels)

        self.mapview = ndimage.zoom(self.mapview,  zoom_params, order=order)    #retains dtype
        self.d, self.dimensions = self.data_from_mapview(self.mapview)
        self.shape = self.d.shape

        self.check()


#meta-class with set of DataSeries
class DataSet:
    """
    meta-class with paired objects for data (int), error (float)

    handles crop, zoom etc functions while maintaining error statistics
    """
    def __init__(self, data, se=None, labels=[], guess_se=True):

        if isinstance(data, DataSet):   #can be initialised from instance of own class
                                        #needed for super PixelSet
            self = data
        else:
            #data handling
            if not isinstance(data, DataSeries):
                data = DataSeries(data)
            
            self.data = data
            self.dimensions = data.dimensions
            self.nchan = self.data.d.shape[1]

            if not np.issubdtype(self.data.d.dtype, np.number):
                raise ValueError('creating a DataSet requires numerical data')

            #label handling
            #   check shape of labels, if given
            if not labels == []:
                self.apply_labels(labels)
            else:
                self.labels= []

            #stderr handling
            if se == None:
                #TO-DO:
                #   may need to adjust this function so se can be truly null
                #   otherwise big datasets without errors will create a huge array of useless zeros in memory 
                if guess_se==True:  
                    self.se = DataSeries(np.sqrt(self.data.d), self.data.dimensions) 
                else:
                    self.se = DataSeries(np.zeros(self.data.d.shape, dtype=np.float32), self.data.dimensions) 
            else:
                if isinstance(se, DataSeries):
                    self.se = se
                else:
                    if len(se.shape) == 3:
                        self.se = DataSeries(se)
                    else:
                        if se.shape == self.data.shape:
                            self.se = DataSeries(se, dimensions=self.data.dimensions)
                        else:
                            raise ValueError('standard error must be 3D map ie. (Y, X, N) OR match data dimensions')

                if not np.issubdtype(self.se.d.dtype, np.number):
                    raise ValueError('standard error must be numerical')

                if not self.se.dimensions == self.data.dimensions:
                    self.match_se_to_data()

            self.weights = np.ones(self.data.d.shape[1], dtype=np.float32)

        self.check()


    def match_se_to_data(self, scale_axis=1):
        yfactor = self.data.dimensions[0] / self.se.dimensions[0]
        xfactor = self.data.dimensions[1] / self.se.dimensions[1]
        if not yfactor == xfactor:
            print(f"WARNING: different ratios for x and y, scaling on axis {scale_axis}, other will be cropped")

        if scale_axis==1:
            zoom_factor = xfactor
            crop_axis=0
        elif scale_axis==0:
            zoom_factor = yfactor
            crop_axis=1
        else:
            raise ValueError("invalid axis given, must be 0 or 1")
        
        self.se.zoom(zoom_factor)

        if self.se.dimensions[crop_axis] > self.data.dimensions[crop_axis]:
            print(f"WARNING: dimensions differ between data and stderr after scaling, cropping on axis {crop_axis}")            
            if crop_axis==1:
                self.se.crop(xrange=(0,self.data.dimensions[crop_axis]))
            else:
                self.se.crop(yrange=(0,self.data.dimensions[crop_axis]))

        self.check()


    def check(self):
        """
        basic sanity checks
        """
        if not self.data.d.shape == self.se.d.shape:
            raise ValueError("shape mismatch between data and serr")        
        
        if not self.nchan == self.data.d.shape[1]:
            raise ValueError("mismatch in no. channels")   

        if not ( ( self.dimensions == self.data.dimensions ) and ( self.dimensions == self.se.dimensions ) ):
            raise ValueError("stored dimension mismatch between data and serr")  

        if not ( self.labels == [] or self.data.d.shape[1] == len(self.labels) ):
            raise ValueError("mismatch between data and label shapes")

        if not np.issubdtype(self.data.d.dtype, np.number):
            raise ValueError("data DataSeries must be numerical")  

        if not np.issubdtype(self.se.d.dtype, np.number):
            raise ValueError("stderr DataSeries must be numerical")    
        
        self.data.check()
        self.se.check()

        return True
    
    def apply_labels(self, labels):
        """
        check and apply a set of labels    
        """
        if len(labels) == self.data.d.shape[1]:
            self.labels = labels
        else:
            raise ValueError("Mismatch between provided labels and data dimensions")

    def resize(self, zoom_factor):
        """
        resize a map, adjusting stderr accordingly  
        """
        if zoom_factor < 1:   
            order = 1   #bicubic for downsampling
            error_factor = np.sqrt(4^order)
        else:
            order = 2   #bilinear for upsampling
            error_factor = 1/np.sqrt(4^order)    #estimate        

        self.data.zoom(zoom_factor, order=order)
        self.se.zoom(zoom_factor, order=order)

        self.se.set_to(self.se.d/error_factor)  #estimate

        self.check()

    def crop(self, xrange=(0, 99999), yrange=(0, 99999)):
        """
        crop maps in 2D and adjustcorresponding 1D view
        """
        self.data.crop(xrange, yrange)
        self.se.crop(xrange, yrange)
        self.check()

    def downsample_by_se(self, deweight=False):

        SD_MULTIPLIER = 2
        DEWEIGHT_FACTOR = 0.5

        self.check()

        if not np.issubdtype(self.data.d.dtype, np.floating):
            print("WARNING: dtype changing to float")

        mapview_ = np.zeros(self.data.mapview.shape, dtype=np.float32)
        se_map_ = np.zeros(self.se.mapview.shape, dtype=np.float32)

        if np.max(self.se.d) == 0:
            print("WARNING: downsampling without valid data for errors - data will be left unchanged")
        else:
            for i in range(self.data.d.shape[1]):

                img_ = np.ndarray.copy(self.data.mapview[:,:,i])
                se_ = np.ndarray.copy(self.se.mapview[:,:,i])

                ratio, q2_sd, q99_data = imgops.calc_quantiles(img_, se_, SD_MULTIPLIER)

                j=0
                while ratio >= 1:
                    print(f"averaging channel {i}, cycle {j} -- dataq99: {q99_data:.3f}, sdq2: {q2_sd:.3f}, ratio: {ratio:.3f}")
                    img_, se_ = imgops.apply_gaussian(img_, 1, se_)

                    if deweight:
                        #deweight channel for each gaussian applied
                        self.weights[i] = self.weights[i]*DEWEIGHT_FACTOR

                    ratio, q2_sd, q99_data = imgops.calc_quantiles(img_, se_, SD_MULTIPLIER)
                    j+=1

                mapview_[:,:,i] = img_
                se_map_[:,:,i] = se_

        self.data.set_to(mapview_)
        self.se.set_to(se_map_)

        self.check()



class PixelSet(DataSet):
    """
    superclass of DataSet with additional methods/attrs

    inherits manually via setattrs, permits creation from instance of subclass
    """
    def __init__(self, dataset):
        super(PixelSet, self).__init__(dataset)
        for attr in dir(dataset):
            if not attr.startswith('__'):    
                setattr(self, attr, getattr(dataset, attr))

        self.weighted = None

    def apply_transform_via_weights(self, transform=None):
        if not self.weights.shape[0] == self.data.shape[1]:
                raise ValueError(f"shape mistmatch between weights {self.weights.shape} and data {self.data.shape}")
        
        for i in range(self.data.shape[1]):
            max_ = np.max(self.data.d[:,i])

            if transform == 'sqrt':
                self.weights[i] = self.weights[i]*sqrt(max_)/max_
            
            if transform == 'log':
                self.weights[i] = self.weights[i]*log(max_)/max_
            
            elif transform == None:
                pass  
            else:
                raise ValueError(f"invalue value for transform: {transform}")         
    
    def apply_weights(self):
        result = np.zeros(self.data.shape)

        for i in range(self.data.shape[1]):
            result[:,i] = self.data.d[:,i]*self.weights[i]
        
        self.weighted = DataSeries(result, self.data.dimensions)

    def apply_transform(self, transform=None):

        if self.weighted == None:
            raise ValueError("PixelSet instance.weighted not initialised")

        if transform == 'sqrt':
            self.weighted.set_to(np.sqrt(self.weighted.d))

        elif transform == 'log':
            self.weighted.set_to(np.log(self.weighted.d))          

        elif transform == None:
            pass  
        else:
            raise ValueError(f"invalue value for transform: {transform}")