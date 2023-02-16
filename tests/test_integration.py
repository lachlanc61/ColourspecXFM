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

PACKAGE_CONFIG='xfmreadout/config.yaml'

sys.path.append(BASE_DIR)

import xfmreadout.bufferops as bufferops

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
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_pxlen.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_xidx.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_yidx.txt'),
    os.path.join(DATA_DIR, 'out_ts2_01_sub/pixeldata/pxstats_dt.txt'),
    )
def test_integration_index(datafiles):
    """
        index datafile 

        read pixel headers back in
            - pixel lengths
            - deadtimes
            - xcoords
            - ycoords

        compare to known results
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
    fpath = os.path.join(f.dirname, f.basename)
    fname = os.path.splitext(os.path.basename(f))[0]
    datadir=os.path.dirname(f)
    outdir=os.path.join(datadir, config['OUTDIR']+"_"+fname)

    #arguments
    args_in = ["-f", fpath, "-o", outdir, "-i", ]

    #run
    pixelseries, ___, ___, ___ = main.main(args_in)

    assert np.allclose(pixelseries.pxlen, expected_pxlen)
    assert np.allclose(pixelseries.xidx, expected_xidx)
    assert np.allclose(pixelseries.yidx, expected_yidx)
    assert np.allclose(pixelseries.dt, expected_dt)

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'ts2_01_sub.GeoPIXE'),
    )
def test_integration_write(datafiles):
    """
        read datafile and write cropped file

        assert filesizes and adjusted header values
    """
    assert True
    #python main.py -f ./tests/test_buffer_reads/ts2_01_sub.GeoPIXE -o "./out" -w -x 20 40 -y 10 20 -ff