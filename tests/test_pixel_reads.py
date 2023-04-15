import pytest
import sys, os
import yaml
import numpy as np

TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR, ___ = os.path.splitext(__file__)


PACKAGE_CONFIG='xfmreadout/config.yaml'

sys.path.append(BASE_DIR)

import tests.utils_tests as ut
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
    os.path.join(DATA_DIR, 'px4398_f_header.bin'),
    )
def test_readpxheader_standard_flat(datafiles):
    """
    regular pixel header from single-detector, no-deadtime format
    """
    f = ut.findin("header.bin", datafiles)
    with open(f, mode='rb') as fi:
        stream = fi.read(PXHEADERLEN)

        expected = [4880, 46, 17, 0, 0.0] 

        pxlen, xidx, yidx, det, dt = bufferops.readpxheader(stream)

        result = [pxlen, xidx, yidx, det, dt]

        assert result == expected


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'px14387_0_header.bin'),
    os.path.join(DATA_DIR, 'px14387_1_header.bin'),
    )
def test_readpxheader_standard_det01(datafiles):
    """
    regular pixel header from two-detector, deadtime-inclusive format

    """             
    expected =  [[ 3868, 51, 56, 0, 15.940695762634277 ], \
                [ 4040, 51, 56, 1, 12.854915618896484 ] ]
    i=0
    for f in datafiles.iterdir():
        with open(f, mode='rb') as fi:
            stream = fi.read(PXHEADERLEN)

            pxlen, xidx, yidx, det, dt = bufferops.readpxheader(stream)

            result = [ pxlen, xidx, yidx, det, dt ]

            assert result == expected[i]
        i+=1


@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'px4398_f_data.bin'),
    #os.path.join(DATA_DIR, 'px4398_f_data.npy'),
    )
def test_readpxdata_standard_flat(datafiles):

    """
    #regular pixel data from single-detector format
    """

    assert 1



@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'px14387_1_data.bin'),
    os.path.join(DATA_DIR, 'px14387_1_data_chan.npy'),
    os.path.join(DATA_DIR, 'px14387_1_data_counts.npy'),
    )
def test_readpxdata_standard_det01(datafiles):
    """
    regular pixel data from two-detector format
    """
    for f in datafiles.iterdir():
        #miserable hack here 
        # - pytest.mark.datafiles not loading files in consistent order
        #   have to search array for addendum (ie. after last underscore)
        #   and assign that way....
        fpath, ext = os.path.splitext(str(f))
        addendum = fpath.split('_')[-1]
        if addendum == 'counts':
            expected_counts = np.load(str(f))         
        elif addendum == 'chan':
            expected_chan = np.load(str(f))   
        elif addendum == 'data':
            with open(f, mode='rb') as fi:
                stream = fi.read() 

    chan, counts = bufferops.readpxdata(stream, len(stream), BYTESPERCHAN, NCHAN)

    assert np.array_equal(chan, expected_chan)
    assert np.array_equal(counts, expected_counts)


def test_readpxdata_empty():
    """
    pixel data empty - should return empty without error
    """
    data = ''
    stream = data.encode(CHARENCODE)
    pixel_length=0

    stream = data.encode(CHARENCODE)
    #expected_chan = []
    expected_chan = np.arange(NCHAN,dtype=np.uint16)
    #expected_counts = []
    expected_counts = np.zeros(NCHAN,dtype=np.uint16)

    chan, counts = bufferops.readpxdata(stream, pixel_length, BYTESPERCHAN, NCHAN)

    assert np.array_equal(chan, expected_chan)
    assert np.array_equal(counts, expected_counts)





























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
    tf = open('/home/lachlan/CODEBASE/ReadoutXFM/binout.bin', 'wb')
    tf.write(stream)
    tf.close()

    #mv with bash



    np.save('/home/lachlan/CODEBASE/ReadoutXFM/binout',counts)

    reload = np.load('/home/lachlan/CODEBASE/ReadoutXFM/binout.npdat')

"""