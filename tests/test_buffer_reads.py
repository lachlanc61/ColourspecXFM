import pytest
import sys, os
import yaml


TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR_NAME="test_data"   #hardcoded for tests dependent on large datafiles
DATA_DIR = os.path.join(TEST_DIR, DATA_DIR_NAME)  

PACKAGE_CONFIG='xfmkit/config.yaml'

sys.path.append(BASE_DIR)

import xfmkit.bufferops as bufferops
import tests.utils_tests as ut

#get config
with open(os.path.join(BASE_DIR, PACKAGE_CONFIG), "r") as f:
    config = yaml.safe_load(f)

#assign constants from config
PXHEADERLEN=config['PXHEADERLEN']
CHARENCODE=config['CHARENCODE']
NCHAN=config['NCHAN']
BYTESPERCHAN=config['BYTESPERCHAN']
MBCONV=config['MBCONV']

"""
@pytest.fixture()
def buffer(infile, chunksize):   
    #infile.seek(0)
    return bufferops.MapBuffer(infile, chunksize)
"""

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_f_sub.GeoPIXE'),
    )
def test_buffer_flat_load(datafiles):
    """
    validate buffer load & retrieval
        single-detector format
    tests single-process and multiprocess

    """
    #Future: pull first ~20 bytes of each chunk and also check those

    chunksize=int(3*MBCONV) #DO NOT MODIFY, affects expected
    expected = \
        [[ 0, 3145728, 3145728 ], \
        [ 3145728, 3145728, 3145728 ], \
        [ 6291456, 2671773, 3145728 ], \
        [ 8963229, 0, 3145728]] #EOF
    #      b.fidx, b.len, b.chunksize

    for multiproc in [ True, False]:
        f = ut.findin("sub.GeoPIXE", datafiles)
        with open(f, mode='rb') as fi:
            buffer=bufferops.MapBuffer(fi, chunksize, multiproc)

            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[0]
            buffer=buffer.retrieve()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[1]
            buffer=buffer.retrieve()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[2]
            buffer=buffer.retrieve()
            buffer.wait()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[3]


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    )
def test_buffer_01_load(datafiles):
    """
    validate buffer load & retrieval via multiprocess
        dual-detector format
    tests single-process and multiprocess
    """
    chunksize=int(5*MBCONV) #DO NOT MODIFY, affects expected
    expected = \
        [[ 0, 5242880, 5242880 ], \
        [ 5242880, 5242880, 5242880 ], \
        [ 10485760, 2223685, 5242880 ], \
        [ 12709445, 0, 5242880]] #EOF
        #     b.fidx, b.len, b.chunksize

    for multiproc in [ True, False]:
        f = ut.findin("sub.GeoPIXE", datafiles)
        with open(f, mode='rb') as fi:
            buffer=bufferops.MapBuffer(fi, chunksize, multiproc)

            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[0]
            buffer=buffer.retrieve()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[1]
            buffer=buffer.retrieve()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[2]
            buffer=buffer.retrieve()
            buffer.wait()
            assert [ buffer.fidx, buffer.len, buffer.chunksize ] == expected[3]

        