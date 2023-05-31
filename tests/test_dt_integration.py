import pytest
import sys, os
import yaml
import numpy as np

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)

BIGDATA_DIR_NAME="test_data"   #hardcoded for tests dependent on large datafiles
BIGDATA_DIR = os.path.join(TEST_DIR, BIGDATA_DIR_NAME)  
DATA_DIR, ___ = os.path.splitext(__file__)

CHUNK_SIZE=5
CONTROL_ARGS=[ "-s", str(CHUNK_SIZE),]
CONTROL_ARGS_MULTIPROC=[ "-s", str(CHUNK_SIZE), "-m"]

PACKAGE_CONFIG='xfmreadout/config.yaml'

sys.path.append(BASE_DIR)

import xfmreadout.bufferops as bufferops
import tests.utils_tests as ut
import xfmreadout.entrypoints as entrypoints


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
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_data.npy'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_dt.npy'),
    )
def test_cycle_dtfill_cpp(datafiles):
    """
        read->write->read cycle:
            writes a .GeoPIXE file with fixed deadtimes, 
            then parses this to confirm data and dt unchanged

        - read datafile and write new file
        - assert filesize is correct
        - read new file back in
        - assert pixeldata and dts are correct

    """
    NEW_DEADTIME=float(33)
    control_args = CONTROL_ARGS_MULTIPROC

    #get expected
    ef = ut.findin("ts2_01_sub_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))     

    ef = ut.findin("dt.npy", datafiles)
    old_dt = np.load(ef) 
    expected_dt = np.full(old_dt.shape,NEW_DEADTIME)

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments for write
    args_in = [ "-f", str(f), "-i", "-dt", str(NEW_DEADTIME), "-w", ] + control_args

    #run crop/write
    ___, ___ = entrypoints.read_raw(args_in)

    #use output from crop/write as next input
    f_result = os.path.join(os.path.dirname(f), "out_ts2_01_sub/ts2_01_sub_mod.GeoPIXE")

    #use output file as input for next run
    next_args_in = [ "-f", f_result, ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(next_args_in)

    #check results
    assert np.allclose(pixelseries.data, expected_pxdata)
    assert np.allclose(pixelseries.dt, expected_dt)