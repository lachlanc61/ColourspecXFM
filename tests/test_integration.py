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
    os.path.join(DATA_DIR, 'ts2_01_sub_pxlen.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_xidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_yidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_dt.npy'),
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
    control_args = CONTROL_ARGS
    #get expected
    ef = ut.findin("pxlen.npy", datafiles)
    expected_pxlen = np.load(ef)
    ef = ut.findin("xidx.npy", datafiles)
    #expected_xidx = np.loadtxt(ef, dtype=np.uint16, delimiter=",")
    expected_xidx = np.load(ef)
    ef = ut.findin("yidx.npy", datafiles)
    expected_yidx = np.load(ef)
    ef = ut.findin("dt.npy", datafiles)
    expected_dt = np.load(ef)

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = ["-f", str(f), "-i", ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(args_in)

    #assert 0
    assert np.allclose(pixelseries.pxlen, expected_pxlen)
    assert np.allclose(pixelseries.xidx, expected_xidx)
    assert np.allclose(pixelseries.yidx, expected_yidx)
    assert np.allclose(pixelseries.dt, expected_dt)

@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_data.npy'),
    )
def test_integration_parse(datafiles):
    """
        parse datafile 

        NB: failing at pxidx 1492, channel 0
        - likely first pixel after new chunk
    """
    control_args = CONTROL_ARGS

    #get expected
    ef = ut.findin("ts2_01_sub_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))       
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = [ "-f", str(f), ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(args_in)

    assert np.allclose(pixelseries.data, expected_pxdata)


@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_export_data.npy')
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
    control_args = CONTROL_ARGS

    expected_size = int(1407157)    #size for written, cropped .GeoPIXE file

    #get expected
    ef = ut.findin("ts2_01_sub_export_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))   
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments for crop/write
    args_in = [ "-f", str(f), "-i", "-w", "-x", "20", "40", "-y", "10", "20", ] + control_args

    #run crop/write
    ___, ___ = entrypoints.read_raw(args_in)

    #use output from crop/write as next input
    f_result = os.path.join(os.path.dirname(f), "out_ts2_01_sub/ts2_01_sub_mod.GeoPIXE")

    #check filesize is correct
    assert os.path.getsize(f_result) == expected_size

    #use output file as input for next run
    next_args_in = [ "-f", f_result, ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(next_args_in)

    #check results
    assert np.allclose(pixelseries.data, expected_pxdata)

#-------------------------------------------------------------------
#------------CPP PARSE----------------------------------------------
#-------------------------------------------------------------------

@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(DATA_DIR, 'ts2_01_sub_pxlen.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_xidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_yidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_dt.npy'),
    )
def test_integration_index_cpp(datafiles):
    """
        index datafile 

        compare to known:
            - pixel lengths
            - deadtimes
            - xcoords
            - ycoords
    """
    control_args = CONTROL_ARGS_MULTIPROC
    #get expected
    ef = ut.findin("pxlen.npy", datafiles)
    expected_pxlen = np.load(ef)
    ef = ut.findin("xidx.npy", datafiles)
    #expected_xidx = np.loadtxt(ef, dtype=np.uint16, delimiter=",")
    expected_xidx = np.load(ef)
    ef = ut.findin("yidx.npy", datafiles)
    expected_yidx = np.load(ef)
    ef = ut.findin("dt.npy", datafiles)
    expected_dt = np.load(ef)

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = ["-f", str(f), "-i", ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(args_in)

    #assert 0
    assert np.allclose(pixelseries.pxlen, expected_pxlen)
    assert np.allclose(pixelseries.xidx, expected_xidx)
    assert np.allclose(pixelseries.yidx, expected_yidx)
    assert np.allclose(pixelseries.dt, expected_dt)

@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_data.npy'),
    )
def test_integration_parse_cpp(datafiles):
    """
        parse datafile 

        NB: failing at pxidx 1492, channel 0
        - likely first pixel after new chunk
    """
    control_args = CONTROL_ARGS_MULTIPROC

    #get expected
    ef = ut.findin("ts2_01_sub_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))       
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments
    args_in = [ "-f", str(f), ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(args_in)

    assert np.allclose(pixelseries.data, expected_pxdata)


@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_export_data.npy')
    )
def test_integration_cycle_cpp(datafiles):
    """
        full read->write->read cycle:
            writes a cropped .GeoPIXE file, then parses this new cropped file and confirms data

        - read datafile and write cropped file
        - assert filesize is correct
        - read cropped output file back in
        - assert parsed pixel array is correct

        FUTURE: assert header values for cropped file
    """
    control_args = CONTROL_ARGS_MULTIPROC

    expected_size = int(1407157)    #size for written, cropped .GeoPIXE file

    #get expected
    ef = ut.findin("ts2_01_sub_export_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))   
    #   np.load seems to need str path while np.loadtxt doesn't

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments for crop/write
    args_in = [ "-f", str(f), "-i", "-w", "-x", "20", "40", "-y", "10", "20", ] + control_args

    #run crop/write
    ___, ___ = entrypoints.read_raw(args_in)

    #use output from crop/write as next input
    f_result = os.path.join(os.path.dirname(f), "out_ts2_01_sub/ts2_01_sub_mod.GeoPIXE")

    #check filesize is correct
    assert os.path.getsize(f_result) == expected_size

    #use output file as input for next run
    next_args_in = [ "-f", f_result, ] + control_args

    #run
    pixelseries, ___ = entrypoints.read_raw(next_args_in)

    #check results
    assert np.allclose(pixelseries.data, expected_pxdata)


@pytest.mark.datafiles(
    os.path.join(BIGDATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(BIGDATA_DIR, 'ts2_01_sub_data.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_pxlen.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_xidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_yidx.npy'),
    os.path.join(DATA_DIR, 'ts2_01_sub_dt.npy'),
    )
def test_cycle_unchanged_cpp(datafiles):
    """
        read->write->read cycle:
            writes an unchanged .GeoPIXE file, then parses this 
            to confirm data and headervals unchanged

        - read datafile and write new file
        - assert filesize is correct
        - read new file back in
        - assert pixeldata and headervals are correct

    """
    control_args = CONTROL_ARGS_MULTIPROC

    #get expected
    ef = ut.findin("ts2_01_sub_data.npy", datafiles)
    expected_pxdata = np.load(str(ef))     
    
    ef = ut.findin("pxlen.npy", datafiles)
    expected_pxlen = np.load(ef)

    ef = ut.findin("xidx.npy", datafiles)
    expected_xidx = np.load(ef)

    ef = ut.findin("yidx.npy", datafiles)
    expected_yidx = np.load(ef)

    ef = ut.findin("dt.npy", datafiles)
    expected_dt = np.load(ef) 

    #prep
    f = ut.findin("ts2_01_sub.GeoPIXE", datafiles)

    #arguments for write
    args_in = [ "-f", str(f), "-i", "-w", ] + control_args

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
    assert np.allclose(pixelseries.pxlen, expected_pxlen)
    assert np.allclose(pixelseries.xidx, expected_xidx)
    assert np.allclose(pixelseries.yidx, expected_yidx)
    assert np.allclose(pixelseries.dt, expected_dt)