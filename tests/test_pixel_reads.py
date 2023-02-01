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
    os.path.join(DATA_DIR, 'px14387_0_header.bin'),
    os.path.join(DATA_DIR, 'px14387_1_data.bin'),
    )
def test_datafile_multiple(datafiles):
    """
    #confirm pytest-datafiles correctly pulls files
    """
    for f in datafiles.listdir():
        assert os.path.isfile(f)



@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'px4398_f_header.bin'),
    )
def test_readpxheader_standard_flat(datafiles):
    """
    regular pixel header from single-detector, no-deadtime format
    """
    f = datafiles.listdir()[0]
    fi = open(f, mode='rb')
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
    for f in datafiles.listdir():
        fi = open(f, mode='rb')
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
    os.path.join(DATA_DIR, 'px14387_1_data_counts.npy'),
    os.path.join(DATA_DIR, 'px14387_1_data_chan.npy'),
    )
def test_readpxdata_standard_det01(datafiles):
    """
    regular pixel data from two-detector format
    """
    for f in datafiles.listdir():
        #miserable hack here 
        # - pytest.mark.datafiles not loading files in consistent order
        #   have to search array for addendum (ie. after last underscore)
        #   and check that way....
        fpath, ext = os.path.splitext(str(f))
        addendum = fpath.split('_')[-1]
        if addendum == 'counts':
            expected_counts = np.load(str(f))         
        elif addendum == 'chan':
            expected_chan = np.load(str(f))   
        elif addendum == 'data':
            fi = open(f, mode='rb')
            stream = fi.read() 

    """
    should be this:
    
    #get expected results
    f_chan = str(datafiles.listdir()[0])
    expected_chan = np.load(f_chan)

    f_counts = str(datafiles.listdir()[1])
    expected_counts = np.load(f_counts)

    #get input data
    f = datafiles.listdir()[2]
    fi = open(f, mode='rb')
    stream = fi.read()
    """

    chan, counts = bufferops.readpxdata(stream, len(stream), BYTESPERCHAN)

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
    tf = open('/home/lachlan/CODEBASE/ReadoutXFM/binout.bin', 'wb')
    tf.write(stream)
    tf.close()

    #mv with bash



    np.save('/home/lachlan/CODEBASE/ReadoutXFM/binout',counts)

    reload = np.load('/home/lachlan/CODEBASE/ReadoutXFM/binout.npdat')

"""