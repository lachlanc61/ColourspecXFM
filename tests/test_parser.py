import pytest
import sys, os

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
#https://stackoverflow.com/questions/25827160/importing-correctly-with-pytest
TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
sys.path.append(BASE_DIR)

import xfmreadout.structures as structures
import xfmreadout.utils as utils
import xfmreadout.bufferops as parser

USER_CONFIG=os.path.join(TEST_DIR,'parser_config.yaml')
PACKAGE_CONFIG=os.path.join(TEST_DIR,'parser_protocol.yaml')

#get command line arguments
args = utils.readargs()

#create input config from args and config files
config, rawconfig=utils.initcfg(args, PACKAGE_CONFIG, USER_CONFIG)

#initialise read file and all directories relative to current script
config, fi, fname, fsub, odir = utils.initf(config)

#define the map object fixture
@pytest.fixture
def xfmap():   
    return structures.Xfmap(config, fi, fsub)

def test_stream_break_body(xfmap):
    """
    check the stream when read across chunk break in pixel body

    NB: test depends on correctly initialising object state at end of chunk
    """
    #values for pixel body at end of 5 Mb chunk
    expected_datafile=os.path.join(TEST_DIR, 'data/endpxcontent.bin')
    xfmap.idx=5242445  #start of pixel body
    xfmap.pxidx=1066
    readlength=4864
    
    #read expected data from file
    expdata = open(expected_datafile, mode='rb')
    expected_length = 4864  #length of testdata
    expected = expdata.read(expected_length)

    #run getstream
    result, xfmap.idx = bufferops.getstream(xfmap,xfmap.idx,readlength)

    assert result == expected

def test_stream_break_header(xfmap):
    """
    check the stream when read across chunk break in pixel header

    NB: test depends on correctly initialising object state at end of chunk
    """
    #set xfmap pointers to pixel header near end of 10Mb chunk
    expected_datafile=os.path.join(TEST_DIR, 'data/endpxheader.bin')
    xfmap.idx=5242429  #start of pixel header
    stepforward=6   #position of break
    xfmap.pxidx=1066

    #truncate stream/chunk to end in middle of header
    xfmap.chunksize=xfmap.idx+stepforward           #set chunksize to header start + step
    xfmap.stream=xfmap.stream[0:xfmap.chunksize]    #drop end of stream
    xfmap.streamlen=len(xfmap.stream)               #assign new length of stream
    xfmap.infile.seek(xfmap.chunksize, 0)           #move file pointer to match new chunk

    #read expected data from file
    expdata = open(expected_datafile, mode='rb')
    expected = expdata.read(xfmap.PXHEADERLEN)

    #run getstream
    result, xfmap.idx = bufferops.getstream(xfmap,xfmap.idx,xfmap.PXHEADERLEN)

    assert result == expected



def test_read_pxheader(xfmap):
    """
    tests pixel header read
    """
    expected = [4880, 46, 17, 0, 0.0] #known result for this pixel

    #open and read the demo pixel header
    f=os.path.join(TEST_DIR, 'data/endpxheader.bin')
    fi = open(f, mode='rb')
    teststream = fi.read(xfmap.PXHEADERLEN)

    #run
    pxlen, xidx, yidx, det, dt = bufferops.readpxheader(teststream, config, xfmap.PXHEADERLEN, xfmap)

    #format as list
    result = [pxlen, xidx, yidx, det, dt]

    assert result == expected


def EXCLUDEtest_read_pxdata(xfmap):
    """
    not working yet

    just need to save counts to file and open, thenc ompare
    may want to change readpxdata so it outputs np.arrays instead of lists
    """
    pxlen=4880
    #open and read the demo pixel header
    f=os.path.join(TEST_DIR, 'data/endpxcontent.bin')
    fi = open(f, mode='rb')
    teststream = fi.read(pxlen)

    expected_datafile=os.path.join(TEST_DIR, 'data/pxchan.csv')

    #check that teststream matches expected pixel length
    #   if fails, test environment not valid
    if not len(teststream) == pxlen-xfmap.PXHEADERLEN:
        raise ValueError("TEST STATE ERROR: unexpected length of input stream")


    chan, counts = bufferops.readpxdata(teststream, config, pxlen-xfmap.PXHEADERLEN)
    
    breakpoint()

    assert counts == exp_counts

"""
def test_gapfill():

    #in_chan=readcsv
    #in_chan=readcsv
    #exp_chan=readcsv
    #exp_counts=readcsv

    chan, counts = utils.gapfill(in_chan,in_counts, config['NCHAN'])

    assert counts == exp_counts


test_read_pxheader(xfmap)

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