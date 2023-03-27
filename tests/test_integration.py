import pytest
import sys, os
import yaml
import numpy as np

import main
import tests.utils_tests as ut

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR_NAME="test_data"   #hardcoded for tests dependent on large datafiles
DATA_DIR = os.path.join(TEST_DIR, DATA_DIR_NAME)  

CHUNK_SIZE=5
CONTROL_ARGS=[ "-s", str(CHUNK_SIZE),]

PACKAGE_CONFIG='xfmreadout/config.yaml'

sys.path.append(BASE_DIR)

import xfmreadout.bufferops as bufferops

#get config
with open(os.path.join(BASE_DIR, PACKAGE_CONFIG), "r") as f:
    config = yaml.safe_load(f)

#assign constants from config
#PXHEADERLEN=config['PXHEADERLEN']
#CHARENCODE=config['CHARENCODE']
#NCHAN=config['NCHAN']
#BYTESPERCHAN=config['BYTESPERCHAN']
#MBCONV=config['MBCONV']


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_pxlen.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_xidx.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_yidx.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_dt.txt'),
    )
def test_integration_index(datafiles):
    """
        index datafile 

        compare to known:
            - pixel lengths
            - deadtimes
            - xcoords
            - ycoords
    """
    #get expected
    ef = ut.findin("pxlen.txt", datafiles)
    expected_pxlen = np.loadtxt(ef, dtype=np.uint16, delimiter=",")
    ef = ut.findin("xidx.txt", datafiles)
    expected_xidx = np.loadtxt(ef, dtype=np.uint16, delimiter=",")
    ef = ut.findin("yidx.txt", datafiles)
    expected_yidx = np.loadtxt(ef, dtype=np.uint16, delimiter=",")
    ef = ut.findin("dt.txt", datafiles)
    expected_dt = np.loadtxt(ef, dtype=np.float32, delimiter=",")

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = ["-f", f.strpath, "-i", ] + CONTROL_ARGS

    #run
    pixelseries, ___ = main.main(args_in)

    assert np.allclose(pixelseries.pxlen, expected_pxlen)
    assert np.allclose(pixelseries.xidx, expected_xidx)
    assert np.allclose(pixelseries.yidx, expected_yidx)
    assert np.allclose(pixelseries.dt, expected_dt)

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxspec.npy'),
    )
def test_integration_parse(datafiles):
    """
        parse datafile 

        NB: failing at pxidx 1492, channel 0
        - likely first pixel after new chunk
    """
    #get expected
    ef = ut.findin("pxspec.npy", datafiles)
    expected_pxdata = np.load(ef.strpath)       
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = [ "-f", f.strpath, ] + CONTROL_ARGS

    #run
    pixelseries, ___ = main.main(args_in)

    assert np.allclose(pixelseries.data, expected_pxdata)


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub_export/pixeldata/pxspec.npy')
    )
def test_integration_cycle(datafiles):
    """
        full read->write->read cycle:
            writes a cropped .GeoPIXE file, then parses this new cropped file and confirms data

        - read datafile and write cropped file
        - assert filesize is correct
        - read cropped output file back in
        - assert parsed pixel array is correct

        FUTURE: assert header values for cropped file
    """
    expected_size = int(1407157)    #size for written, cropped .GeoPIXE file

    #get expected
    ef = ut.findin("pxspec.npy", datafiles)
    expected_pxdata = np.load(ef.strpath)   
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments for crop/write
    args_in = [ "-f", f.strpath, "-i", "-w", "-x", "20", "40", "-y", "10", "20", ] + CONTROL_ARGS

    #run crop/write
    ___, ___ = main.main(args_in)

    #use output from crop/write as next input
    f_result = os.path.join(f.dirname, "out_ts2_01_sub/ts2_01_sub_export.GeoPIXE")

    #check filesize is correct
    assert os.path.getsize(f_result) == expected_size

    #use output file as input for next run
    next_args_in = [ "-f", f_result, ] + CONTROL_ARGS

    #run
    pixelseries, ___ = main.main(next_args_in)

    #check results
    assert np.allclose(pixelseries.data, expected_pxdata)