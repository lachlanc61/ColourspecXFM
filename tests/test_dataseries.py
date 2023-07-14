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

    ds = structures.DataSeries(array, dimensions=DIMENSIONS)

    ds.data[25,1] = 1.0

    assert ds.mapview[2,5,1] == 1.0

    ds.mapview[2,5,2] = 1.0

    assert ds.data[25,2] == 1.0    


def test_dataseries_crop():
    SHAPE=(400,20)
    DIMENSIONS=(40,10)
    XCROP=(0,10)
    YCROP=(0,20)

    array = np.zeros(SHAPE, dtype=np.float32)
    ds = structures.DataSeries(array, dimensions=DIMENSIONS)

    ds.crop(xrange=XCROP, yrange=YCROP)    

    ds.mapview[1,1,2]=2.0

    assert ds.data[DIMENSIONS[1]+1,2] == 2.0

    ds.data[DIMENSIONS[1]+1,3] = 2.0

    assert ds.mapview[1,1,3] == 2.0