import pytest
import sys, os
import yaml
import numpy as np

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR_NAME="test_stream_reads"   #hardcoded for tests dependent on large datafiles
DATA_DIR = os.path.join(TEST_DIR, DATA_DIR_NAME)  

PACKAGE_CONFIG='xfmreadout/config.yaml'

sys.path.append(BASE_DIR)

import parsercore
import xfmreadout.bufferops as bufferops
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

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_submap_buffer.bin'),
    os.path.join(DATA_DIR, 'ts2_submap_data.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_indexlist.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_pxlen.npy'),
    )
def test_parsercore_read(datafiles):

    #get expected
    ef = ut.findin("data.npy", datafiles)
    expected = np.load(ef)

    #get input arrays
    ef = ut.findin("indexlist.npy", datafiles)
    indexlist = np.load(ef)
    ef = ut.findin("pxlen.npy", datafiles)
    pxlen = np.load(ef)
    
    #get stream
    ef = ut.findin("buffer.bin", datafiles)
    with open(ef, mode='rb') as efi:
        stream = efi.read() 
    
    result = parsercore.readstream(indexlist, pxlen, stream, len(stream))

    assert np.allclose(result, expected)


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_submap_buffer.bin'),
    os.path.join(DATA_DIR, 'ts2_submap_data.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_indexlist.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_pxlen.npy'),
    )
def test_parsercore_catch_f_contig(datafiles):

    #get input arrays
    ef = ut.findin("indexlist.npy", datafiles)
    indexlist = np.load(ef)
    ef = ut.findin("pxlen.npy", datafiles)
    pxlen = np.load(ef)
    
    #get stream
    ef = ut.findin("buffer.bin", datafiles)
    with open(ef, mode='rb') as efi:
        stream = efi.read() 
    
    #convert to fortran array
    indexlist_f=np.asfortranarray(indexlist)    

    try:
        result = parsercore.readstream(indexlist_f, pxlen, stream, len(stream))

    except RuntimeError as e:
        if "C-contiguous" in str(e):
            print("Correctly identified non-C-contiguous array")
            print(f"     Recieved error: {str(e)}")
            assert 1
        else:
            print("Did not identify non-C-contiguous array")
            print(f"     Recieved error: {str(e)}")            
            assert 0
    

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_submap_buffer.bin'),
    os.path.join(DATA_DIR, 'ts2_submap_data.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_indexlist.npy'),
    os.path.join(DATA_DIR, 'ts2_submap_pxlen.npy'),
    )
def test_parsercore_catch_indexshape_mismatch(datafiles):

    #get expected
    ef = ut.findin("data.npy", datafiles)
    expected = np.load(ef)

    #get input arrays
    ef = ut.findin("indexlist.npy", datafiles)
    indexlist = np.load(ef)
    ef = ut.findin("pxlen.npy", datafiles)
    pxlen = np.load(ef)
    
    #get stream
    ef = ut.findin("buffer.bin", datafiles)
    with open(ef, mode='rb') as efi:
        stream = efi.read() 
    
    try:
        result = parsercore.readstream(indexlist[0:100,:], pxlen, stream, len(stream))

    except RuntimeError as e:
        if "shapes differ" in str(e):
            print("Correctly identified shape mismatch")
            print(f"     Recieved error: {str(e)}")
            assert 1
        else:
            print("Did not catch shape mismatch")
            print(f"     Recieved error: {str(e)}")
            assert 1