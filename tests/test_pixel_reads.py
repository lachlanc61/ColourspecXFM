import pytest
import sys, os
import yaml
import numpy as np

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR, ___ = os.path.splitext(__file__)


PACKAGE_CONFIG='xfmreadout/protocol.yaml'

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



# https://pypi.org/project/pytest-datafiles/#description
# pytest datafiles
# multiple files:
"""
    for file in datafiles.listdir():
        print(file)
        print(os.path.isfile(file))
        assert os.path.isfile(file)
"""

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'endpxheader.bin'),
    os.path.join(DATA_DIR, 'endpxcontent.bin'),
    )
def test_datafile_multiple(datafiles):
    """
    #confirm pytest-datafiles correctly pulls files
    """
    for f in datafiles.listdir():
        assert os.path.isfile(f)



@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'endpxheader.bin'),
    )
def test_readpxheader_standard_det0(datafiles):
    """
    regular pixel header from single-detector, no-deadtime format
    """
    f = datafiles.listdir()[0]
    fi = open(f, mode='rb')
    stream = fi.read(PXHEADERLEN)

    expected = [int(4880), int(46), int(17), int(0), float(0.0)] 

    pxlen, xidx, yidx, det, dt = bufferops.readpxheader(stream)

    result = [pxlen, xidx, yidx, det, dt]

    assert result == expected

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'endpxheader.bin'),
    )
def test_readpxheader_standard_det2(datafiles):
    """
    regular pixel header from two-detector, deadtime-inclusive format

    NB: need to pull new data
    """
    f = datafiles.listdir()[0]
    fi = open(f, mode='rb')
    stream = fi.read(PXHEADERLEN)

    expected = [int(4880), int(46), int(17), int(0), float(0.0)] 

    pxlen, xidx, yidx, det, dt = bufferops.readpxheader(stream)

    result = [pxlen, xidx, yidx, det, dt]

    assert result == expected

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'endpxheader.bin'),
    )
def test_readpxdata_standard_det0(datafiles):
    """
    #regular pixel data from single-detector format
    """
    assert 1

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'endpxheader.bin'),
    )
def test_readpxdata_standard_det2(datafiles):
    """
    regular pixel data from two-detector format
    """

    assert 1


def test_readpxdata_empty():
    """
    pixel data empty - should return empty without error
    """
    data = ''
    stream = data.encode(CHARENCODE)
    pixel_length=0

    stream = data.encode(CHARENCODE)
    expected_chan = []
    expected_counts = []

    chan, counts = bufferops.readpxdata(stream, pixel_length, BYTESPERCHAN)

    assert ( chan == expected_chan )
    assert ( counts == expected_counts )





























"""
future tests:
    pull single pixel from subts2
        -> save in separate file

    read that pixel as stream
    T pull header and compare to correct vals
    T pull spectrum and compare to reference

    integration:
    
    parse subts2
    -> compare RGB and sum spectrum 
    write 10x10 from subts2
    parse subsubts2
    -> compare RGB and sum spectrum



"""
"""



working with files:
    #full pixel when pointer at start of body:
        self.stream[self.idx-self.PXHEADERLEN:self.idx]
    
    #write stream to file
    tfo=open('tests/data/endpxcontent.bin', mode='wb')
    tfo.write(locstream)

"""