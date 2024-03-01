import pytest
import sys, os
import numpy as np

#PATHS
TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
sys.path.append(BASE_DIR)

#CONFIG REASSIGNMENT
import xfmkit.config as config
CONF_FILE="tests/test_conf/xfmkit.conf"
config.setup(conf_file=CONF_FILE)   #reassign config - may not actually work

#IMPORT LOCAL
import xfmkit.utils as utils
import xfmkit.structures as structures

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



def test_smartcast_ok():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    MIN=0
    MAX=100
    rng = np.random.default_rng()

    dtype_list = [ np.float32, np.float64, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64 ]

    for dtype_origin in dtype_list:
        if np.issubdtype(dtype_origin, np.floating):
            array_origin = rng.random(SHAPE, dtype=dtype_origin)*MAX
        else:
            array_origin = rng.integers(MIN, MAX, SHAPE, dtype=dtype_origin)

        for dtype_new in dtype_list:
            array_new = utils.smartcast(array_origin, dtype_new)

            assert(np.allclose(array_new, array_origin, atol=1))

def test_dataseries_fill_2D():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    MIN=0
    MAX=1000
    rng = np.random.default_rng()

    dtype_list = [ np.float32, np.float64, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64 ]

    for dtype_origin in dtype_list:

        if np.issubdtype(dtype_origin, np.floating):
            array_origin = rng.random(SHAPE, dtype=dtype_origin)*MAX
        else:
            array_origin = rng.integers(MIN, MAX, SHAPE, dtype=dtype_origin)

        for dtype_new in dtype_list:
            data = structures.DataSeries(array_origin, dimensions=DIMENSIONS)

            if np.issubdtype(dtype_new, np.floating):
                array_new = rng.random(SHAPE, dtype=dtype_new)*MAX
            else:
                array_new = rng.integers(MIN, MAX, SHAPE, dtype=dtype_new)

            data.fill_from(array_new)

            assert data.check()
            assert(np.allclose(data.d, array_new, atol=1))

def test_dataseries_fill_3D():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    new_shape = ( DIMENSIONS[0], DIMENSIONS[1], SHAPE[1] )
    MIN=0
    MAX=1000
    rng = np.random.default_rng()

    dtype_list = [ np.float32, np.float64, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64 ]

    for dtype_origin in dtype_list:

        if np.issubdtype(dtype_origin, np.floating):
            array_origin = rng.random(SHAPE, dtype=dtype_origin)*MAX
        else:
            array_origin = rng.integers(MIN, MAX, SHAPE, dtype=dtype_origin)

        for dtype_new in dtype_list:
            data = structures.DataSeries(array_origin, dimensions=DIMENSIONS)

            if np.issubdtype(dtype_new, np.floating):
                array_new = rng.random(new_shape, dtype=dtype_new)*MAX
            else:
                array_new = rng.integers(MIN, MAX, new_shape, dtype=dtype_new)

            data.fill_from(array_new)

            assert data.check()
            assert(np.allclose(data.mapview, array_new, atol=1))
        

def test_dataseries_set_2D():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    MIN=0
    MAX=1000
    rng = np.random.default_rng()

    dtype_list = [ np.float32, np.float64, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64 ]

    for dtype_origin in dtype_list:

        if np.issubdtype(dtype_origin, np.floating):
            array_origin = rng.random(SHAPE, dtype=dtype_origin)*MAX
        else:
            array_origin = rng.integers(MIN, MAX, SHAPE, dtype=dtype_origin)

        for dtype_new in dtype_list:
            data = structures.DataSeries(array_origin, dimensions=DIMENSIONS)

            if np.issubdtype(dtype_new, np.floating):
                array_new = rng.random(SHAPE, dtype=dtype_new)*MAX
            else:
                array_new = rng.integers(MIN, MAX, SHAPE, dtype=dtype_new)

            data.set_to(array_new)

            assert data.check()
            assert(np.allclose(data.d, array_new, atol=1))


def test_dataseries_set_3D():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    new_shape = ( DIMENSIONS[0], DIMENSIONS[1], SHAPE[1] )
    MIN=0
    MAX=1000
    rng = np.random.default_rng()

    dtype_list = [ np.float32, np.float64, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64 ]

    for dtype_origin in dtype_list:

        if np.issubdtype(dtype_origin, np.floating):
            array_origin = rng.random(SHAPE, dtype=dtype_origin)*MAX
        else:
            array_origin = rng.integers(MIN, MAX, SHAPE, dtype=dtype_origin)

        for dtype_new in dtype_list:
            data = structures.DataSeries(array_origin, dimensions=DIMENSIONS)

            if np.issubdtype(dtype_new, np.floating):
                array_new = rng.random(new_shape, dtype=dtype_new)*MAX
            else:
                array_new = rng.integers(MIN, MAX, new_shape, dtype=dtype_new)

            data.set_to(array_new)

            assert data.check()
            assert(np.allclose(data.mapview, array_new, atol=1))



def test_pxset_create():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)

    array_d = np.random.randint(0, 100, SHAPE)
    array_se = np.random.uniform(0.0, 10.0, SHAPE)

    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)
    stderr = structures.DataSeries(array_se, dimensions=DIMENSIONS)    

    ds = structures.DataSet(data, stderr)

    px = structures.PixelSet(ds)

    assert px.check()


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

    pxs = structures.PixelSet(ds)

    pxs.downsample_by_se()

    assert pxs.check()


def test_dataset_downsample_without_se():
    """
    check behaviour if no stderror given

    downsample_by_se should compelte but leave unchanged
    """
    SHAPE=(400,20)
    DIMENSIONS=(40,10)

    array_d = np.random.randint(0, 100, SHAPE)          #int

    #assert 0
    data = structures.DataSeries(array_d, dimensions=DIMENSIONS)

    #assert 0
    ds = structures.DataSet(data)
    expected = ds.data.d

    pxs = structures.PixelSet(ds)

    pxs.downsample_by_se()

    assert pxs.check()
    assert np.allclose(pxs.data.d, expected)

#    assert ds.data.d.shape == ds.se.d.shape
#    assert ds.data.dimensions == ds.se.dimensions