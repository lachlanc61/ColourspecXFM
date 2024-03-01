import pytest
import sys, os
import numpy as np

#PATHS
TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
DATA_DIR, ___ = os.path.splitext(__file__)
sys.path.append(BASE_DIR)

#CONFIG REASSIGNMENT
import xfmkit.config as config
CONF_FILE="tests/test_conf/xfmkit.conf"
config.setup(conf_file=CONF_FILE)   #reassign config

#IMPORT LOCAL
import tests.utils_tests as ut
import xfmkit.entry_processed as entry_processed

CONTROL_ARGS=[ "-ff" ]

@pytest.mark.datafiles(
    os.path.join(DATA_DIR, 'data.npy'),  
    os.path.join(DATA_DIR, 'testimage-Fe.tiff'),  
    os.path.join(DATA_DIR, 'testimage-Si.tiff'),    
    os.path.join(DATA_DIR, 'testimage-Ca.tiff'),
    os.path.join(DATA_DIR, 'testimage-K.tiff'),
    os.path.join(DATA_DIR, 'testimage-S.tiff'),
    os.path.join(DATA_DIR, 'testimage-Cu.tiff'),
    os.path.join(DATA_DIR, 'testimage-Fe-var.tiff'),  
    os.path.join(DATA_DIR, 'testimage-Si-var.tiff'),    
    os.path.join(DATA_DIR, 'testimage-Ca-var.tiff'),
    os.path.join(DATA_DIR, 'testimage-K-var.tiff'),
    os.path.join(DATA_DIR, 'testimage-S-var.tiff'),
    os.path.join(DATA_DIR, 'testimage-Cu-var.tiff'),    
    )
def test_integration_proc(datafiles):
    """
        classify set of output .tiffs with variance

        confirm number of classes
    """
    control_args = CONTROL_ARGS

    #get expected
    ef = ut.findin("data.npy", datafiles)
    expected_data = np.load(ef)

    f = ut.findin("testimage-Fe.tiff", datafiles)
    d = os.path.dirname(f)

    #arguments
    args_in = ["-d", str(d), ] + control_args

    #run
    pxs, embedding, categories, classavg, palette = entry_processed.read_processed(args_in)

    #assert embedding is largely nonzero
    assert len(np.nonzero(embedding)[0]) > embedding.shape[0]*0.9
    assert len(np.nonzero(embedding)[1]) > embedding.shape[1]*0.9
    #assert number of categories is between 1 and 100
    assert ( np.max(categories) > 1 and np.max(categories) < 100 )
    #assert data is as expected
    assert np.allclose(pxs.data.d, expected_data)