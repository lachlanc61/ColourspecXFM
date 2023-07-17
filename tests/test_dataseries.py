import pytest
import sys, os
import numpy as np

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)

sys.path.append(BASE_DIR)

import xfmreadout.structures as structures

def test_dataseries_mapview():

    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    array = np.zeros(SHAPE, dtype=np.float32)

    data = structures.DataSeries(array, dimensions=DIMENSIONS)

    data.d[25,1] = 1.0

    assert data.mapview[2,5,1] == 1.0

    data.mapview[2,5,2] = 1.0

    assert data.d[25,2] == 1.0    


def test_dataseries_crop():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    XCROP=(0,10)
    YCROP=(0,20)

    array = np.zeros(SHAPE, dtype=np.float32)
    data = structures.DataSeries(array, dimensions=DIMENSIONS)

    data.crop(xrange=XCROP, yrange=YCROP)    

    data.mapview[1,1,2]=2.0

    assert data.d[DIMENSIONS[1]+1,2] == 2.0

    data.d[DIMENSIONS[1]+1,3] = 2.0

    assert data.mapview[1,1,3] == 2.0

def test_dataset_create():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)

    array = np.random.randint(0, 100, SHAPE)
    data = structures.DataSeries(array, dimensions=DIMENSIONS)

    ds = structures.DataSet(data)

    assert np.allclose(ds.data.d, array)

def test_dataset_create_stderr():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)

    array_d = np.random.randint(0, 100, SHAPE)
    array_se = np.random.uniform(0.0, 10.0, SHAPE)

    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=DIMENSIONS)    

    ds = structures.DataSet(data, stderr)

    assert np.allclose(ds.data.d, array_d)
    assert np.allclose(ds.se.d, array_se)


def test_dataset_create_stderr_smaller():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    se_dimensions=(int(DIMENSIONS[0]/2),int(DIMENSIONS[1]/2))
    se_shape=(se_dimensions[0]*se_dimensions[1],SHAPE[1])

    array_d = np.random.randint(0, 100, SHAPE) 
    array_se = np.random.uniform(0.0, 10.0, se_shape) 

    #assert 0
    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=se_dimensions)    

    #assert 0
    ds = structures.DataSet(data, stderr)

    assert np.allclose(ds.data.d, array_d)    
    assert ds.data.d.shape == ds.se.d.shape
    assert ds.data.dimensions == ds.se.dimensions

def test_dataset_create_stderr_larger():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    se_dimensions=(int(DIMENSIONS[0]*2),int(DIMENSIONS[1]*2))
    se_shape=(se_dimensions[0]*se_dimensions[1],SHAPE[1])

    array_d = np.random.randint(0, 100, SHAPE) 
    array_se = np.random.uniform(0.0, 10.0, se_shape) 

    #assert 0
    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=se_dimensions)    

    #assert 0
    ds = structures.DataSet(data, stderr)

    assert np.allclose(ds.data.d, array_d)    
    assert ds.data.d.shape == ds.se.d.shape
    assert ds.data.dimensions == ds.se.dimensions

def test_dataset_create_stderr_diffaxes():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    se_dimensions=(int(DIMENSIONS[0]/2),int(DIMENSIONS[1]/3))   #different scalars
    se_shape=(se_dimensions[0]*se_dimensions[1],SHAPE[1])

    array_d = np.random.randint(0, 100, SHAPE) 
    array_se = np.random.uniform(0.0, 10.0, se_shape) 

    #assert 0
    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=se_dimensions)    

    #assert 0
    ds = structures.DataSet(data, stderr)

    assert np.allclose(ds.data.d, array_d)    
    assert ds.data.d.shape == ds.se.d.shape
    assert ds.data.dimensions == ds.se.dimensions



def test_dataset_downsample_by_se():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    se_dimensions=(int(DIMENSIONS[0]/2),int(DIMENSIONS[1]/3))   #different scalars
    se_shape=(se_dimensions[0]*se_dimensions[1],SHAPE[1])

    array_d = np.random.randint(0, 100, SHAPE)          #int
    array_se = np.random.uniform(0.0, 10.0, se_shape)    #float
    
    #scale se so some channels are high relative to data
    se_scalars = np.random.randint(1,20, SHAPE[1])     
    for i in range(SHAPE[1]):
        array_se[:,i] = array_se[:,i]*se_scalars[i]

    #assert 0
    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=se_dimensions)    

    #assert 0
    ds = structures.DataSet(data, stderr)

    ds.downsample_by_se()

    assert ds.check()

#    assert ds.data.d.shape == ds.se.d.shape
#    assert ds.data.dimensions == ds.se.dimensions