import os
import numpy as np
from scipy import ndimage

import xfmkit.bufferops as bufferops
import xfmkit.dtops as dtops
import xfmkit.imgops as imgops
import xfmkit.utils as utils
import xfmkit.config as config

from ._utils import *

import logging
logger = logging.getLogger(__name__)


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
        crop maps in 2D and adjust corresponding 1D view
        """
        _mapview = self.mapview[yrange[0]:yrange[1], xrange[0]:xrange[1], :]
        self.mapview = np.copy(_mapview)     
        self.d, self.dimensions = self.data_from_mapview(self.mapview)
        self.shape = self.d.shape

        self.check()        

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
    meta-class with paired objects for data (int or float), error (float)

    plus weights per channel

    handles crop, zoom etc functions while maintaining error statistics and weights
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

            #weights tracked here to allow modification with eg. downsampling
            #   used by superclass PixelSeries
            self.weights = np.ones(self.data.d.shape[1], dtype=np.float32)            

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
                    #currently, throw an error if we have no stderr
                    raise ValueError("no standard errors found")
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

        if not self.weights.shape[0] == self.data.d.shape[1]:
            raise ValueError("mismatch between weights and data")  

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
        print("CROPPING")      
        self.data.crop(xrange, yrange)         
        self.se.crop(xrange, yrange)
        self.dimensions = self.data.dimensions    
        self.check()

    

class PixelSet(DataSet):
    """
    superclass of DataSet with additional data transformations

    """

    #from . import _preprocessing

    from ._preprocessing import generate_weighted, apply_direct_transform, weight_by_transform, downsample_by_se, apply_weights

    def __init__(self, dataset):

        #create from instance of subclass
        super(PixelSet, self).__init__(dataset)
        
        #inherit manually via setattrs
        for attr in dir(dataset):
            if not attr.startswith('__'):    
                setattr(self, attr, getattr(dataset, attr))

        self.weighted = None
    