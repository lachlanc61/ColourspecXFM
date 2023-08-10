import time
import sys
import os
import yaml

import numpy as np

from scipy.stats import norm

import logging
logger = logging.getLogger(__name__)

class DirectoryStructure:
    """
    object holding file locations and directory structure
    """
    def __init__(self, args, config):
        """
        Assign input and output directories from the config
            or location of script if relative
        """
        self.script = os.path.realpath(__file__) #_file = current script
        self.spath=os.path.dirname(self.script) 
        self.spath=os.path.dirname(self.spath)
        
        #INPUT FILE
        #check if filepath is absolute based on leading /
        if args.input_file.startswith('/'):
            self.fi=args.input_file
        else:
            self.fi = os.path.join(self.spath,args.input_file)

        #extract name of input file
        self.fname = os.path.splitext(os.path.basename(self.fi))[0]

        #LOGFILE
        if args.log_file is not None:
            if args.log_file.startswith('/'):
                self.logf=args.log_file
            else:
                self.logf = os.path.join(self.spath,args.log_file)
        else:
            self.logf = None

        #assign output:
        #   to input_fname if output blank
        #   otherwise as assigned
        appended_dir=config['OUTDIR']+"_"+self.fname

        if args.output_directory == "" or args.output_directory == None:
            self.odir=os.path.join(os.path.dirname(self.fi),appended_dir)
        elif args.output_directory.startswith('/'):   #relative vs absolute
            self.odir=args.output_directory
        else:
            self.odir=os.path.join(self.spath,args.output_directory)
    
        if self.odir.endswith('/'):
            self.odir = os.path.dirname(self.odir)

        #extract terminal directory for output
        outbase=os.path.basename(self.odir)

        #if terminal directory has the correct format, continue
        #   otherwise, add a subdir with that name
        if outbase == config['OUTDIR'] or outbase == appended_dir:      
            pass
        else:
            self.odir=os.path.join(self.odir,config['OUTDIR'])

        #assign and create analysis subdirs, if needed
        self.embeddings=os.path.join(self.odir,config['EMBED_DIR'])
        self.plots=os.path.join(self.odir,config['PLOTDIR'])
        self.exports=os.path.join(self.odir,config['EXPORTDIR'])

        #setup submap export location and extension
        if args.write_modified:
            self.subname=self.fname+config['write_suffix']
            self.fsub = os.path.join(self.odir,self.subname+config['FTYPE'])

            if not self.subname == os.path.splitext(os.path.basename(self.fsub))[0]:
                raise ValueError(f"submap name not recognisable")

        else:
            self.fsub = None

        return

    def create(self, config):
        """
        create the output directory and subdirectories, if needed
        """

        if not os.path.isdir(self.odir):
            os.mkdir(self.odir)
        
        for dir in [ self.embeddings, self.plots, self.exports ]:
            if not os.path.isdir(dir):
                os.mkdir(dir)

        return self

    def check(self, config):
        """
        run some basic sanity checks
            eg. check for correct filetype
        """
        #check filetype is recognised - currently hardcoded
        if not config['FTYPE'] == ".GeoPIXE":
            raise ValueError(f"Filetype {config['FTYPE']} not recognised")
    
        for dir in [ self.odir, self.embeddings, self.plots, self.exports ]:
                if not os.path.isdir(dir):
                    raise FileNotFoundError(f"Directory {dir} expected but not found")
        
        if not os.path.exists(self.fi):
            raise FileNotFoundError(f"Input file {self.fi} expected but not found")

        return 

    def show(self):
        """
        present the directory assignments to the user
        """
        print(
            "---------------------------\n"
            "PATHS\n"
            "---------------------------\n"
            f"local: {self.spath}\n"
            f"data: {self.fi}\n"
            f"output: {self.odir}"
        )
        if self.fsub != None:
            print(f"submap: {self.fsub}")

        print("---------------------------")
        print("---------------------------")

        return


def readcfg(filename):
    """
    read in the config yaml as a dict
    """
    dir = os.path.realpath(__file__) #_file = current file (ie. utils.py)
    dir=os.path.dirname(dir) 
    dir=os.path.dirname(dir)        #second call to get out of src/..

    yamlfile=os.path.join(dir,filename)

    with open(yamlfile, "r") as f:
        return yaml.safe_load(f)


def initcfg(pkgconfig):
    """
    initialise the config dict
    """
    #parse the config files 
    config=readcfg(pkgconfig) 

    return config


def initfiles(args, config):
    """
    initialise directory object and sanity check it
    """

    dirs = DirectoryStructure(args, config)
    dirs = dirs.create(config)
    dirs.check(config)
    dirs.show()    

    return config, dirs

#    return config, fi, fname, fsub, odir

def lookfor(x, val):
    difference_array = np.absolute(x-val)
    index = difference_array.argmin()
    return index

def normgauss(x, mu, sig1, amp):
    """
    creates a gaussian along x
    normalised so max = amp
    """
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)

def timed(f):
    """

    measures time to run function f
    returns tuple of (output of function), time
        WARNING: not sure what happens when f() itself returns tuple

    call as: 
        out, runtime=timed(lambda: gapfill2(data))
    
    https://stackoverflow.com/questions/5478351/python-time-measure-function
    """
    start = time.time()
    ret = f()
    elapsed = time.time() - start
    return ret, elapsed

def gapfill(x, y, nchannels):
    """
    fills gaps in function using dict
    
    basically assign dict of i,y pairs
        use dict to return default value of (i,0) if i not in dict

        kludge here - we only want (i,0) but *d fails if not given a (0,0) tuple
            .: give (i,(0,0)) but slice out first 0 only
        sure there is a better way to do this
    
    original:
        d = {k: v for k, *v in data}
        return([(i, *d.get(i, (0, 0))) for i in range(nchannels)])

    https://stackoverflow.com/questions/54724987/python-filling-gaps-in-list
    """
    d={}
    j=0
    for k in x:
                d[k] = (y[j],0)
                j+=1
    xout=np.zeros(nchannels,dtype=np.uint16)
    yout=np.zeros(nchannels, dtype=np.uint16)

    for i in range(nchannels):
        xout[i]=i
        yout[i]=(d.get(i, (0, 0))[0])
    return xout, yout

def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,    https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
                if abs(num) < 1024.0:
                        return "%3.1f %s%s" % (num, unit, suffix)
                num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

def varsizes(allitems):
    print(
        "---------------------------\n"
        "Memory usage:\n"
        "---------------------------\n"
    )
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in allitems),
                                                    key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def pxinsubmap(xin, yin, xread, yread):
    if (xread >= xin[0] and xread < xin[1] and
            yread >= yin[0] and yread < yin[1]
    ):
        return True
    else:
        return False


def findelement(elements: list, target:str):
    """
    search a sorted list and return first matching index
    """
    for idx, name in enumerate(elements):
        if name == target:
            return idx

def get_map(data, dims, elements, target: str):
    idx = findelement(elements, target)

    img=map_roll(data[:,idx], dims)

    #DEBUG
    return img


def map_roll(indata, dims, single=False):
    """
    restores map from linear data + map dimensions

    data (n, chan)
    OR
    data (x, y)
    """
    print(indata.shape)
    #if np.shape(indata.shape)[0] == 2:
    if single == True:
        return indata.reshape(dims[0], -1)
    else:        
        return indata.reshape(dims[0], dims[1], -1)
    
def map_unroll(maps):
    """
    reshape map (x, y, counts) to data (i, counts)

    returns dataset and dimensions
    """

    if len(maps.shape) == 3:
        data=maps.reshape(maps.shape[0]*maps.shape[1],-1)
        dims=maps.shape[:2]
    elif len(maps.shape) == 2:
        data=maps.reshape(maps.shape[0]*maps.shape[1])
        dims=maps.shape[:2]        
    else:
        raise ValueError(f"unexpected dimensions for {map}")

    return data, dims

def norm_channel(in_array, new_max=255):
    """
    normalise an array from 0 to new_max
    ie. map to 0-255 for visualisation

    returns an p.uint16 array
    """
    result_ = np.copy(in_array)
    result_ = result_-np.min(result_)
    result_ = (result_/np.max(result_))    
    result_ = np.ndarray.astype(result_*new_max,np.uint16)
    return result_    

def norm_channel_float(in_array, new_max=1.0):
    """
    normalise an array from 0 to new_max
    ie. map to 0-255 for visualisation

    returns an p.uint16 array
    """
    in_array = in_array-np.min(in_array)
    in_array = (in_array/np.max(in_array))    
    in_array = np.ndarray.astype(in_array*new_max,np.float32)
    return in_array  

def count_categories(categories):
    """
    return the total number of categories, including negative values
    """
    min_cat = np.min(categories)

    #  maintain a category for unclassified even if this is empty
    if min_cat > 0:
        min_cat = 0

    max_cat = np.max(categories)
    
    cat_list = range(min_cat, max_cat+1)

    num_cats = max_cat - min_cat + 1

    return num_cats, cat_list

def get_centroid(embedding):
    """
    finds the centroid of a 2D array
    """    
    if len(embedding.shape) != 2:
        raise ValueError("invalid dimensionality for embedding, expected shape == [X, Y]")

    npx = embedding.shape[0]

    result = np.zeros(embedding.shape[1], dtype=np.float32)

    for i in range(embedding.shape[1]):
        result[i] = np.sum(embedding[:, i])

    result = result/npx

    return result

def mean_within_quantile(data, qmin=0.0, qmax=1.0):
    """
    calculate the mean across the values between two quantiles
    """

    qlo = np.quantile(data,qmin)

    qhi = np.quantile(data,qmax)

    mask = np.where(np.logical_and(data >= qlo, data <= qhi ), True, False)

    subset = data[mask]

    result = np.mean(subset)

    return result

def compile_centroids(embedding, categories):
    """
    finds the centroid of each cluster, given an embedding and a categorised array
    """

    FIRST_CATEGORISED=1

    if embedding.shape[0] != categories.shape[0]:
        raise ValueError("Embedding and category list have different number of pixels")

    n_clusters, ___ = count_categories(categories)

    centroids=np.zeros((n_clusters, embedding.shape[1]), dtype=np.float32)

    for i in range(FIRST_CATEGORISED, n_clusters):
        centroids[i] = get_centroid(embedding[categories==i])    

    return centroids


def get_closest_points(embedding, points):
    """
    selects X,Y points from 2D embedding that are closest to each X,Y in points

    returns numpy array of integer indexes corresponding to points in embedding         
    """

    if not ( len(points.shape) == 2 and len(embedding.shape) == 2):
        raise ValueError("both arrays must be 2D")

    indices = np.zeros(points.shape[0], dtype=np.int32)

    for i in range(points.shape[0]):
        matrix = embedding-points[i,:]
        dist = matrix[:,0]**2 + matrix[:,1]**2

        j=0
        partn = np.partition(dist, j)[j]
        result = int(np.where(dist==partn)[0][0])

        while result in indices:
            partn = np.partition(dist, j)[j]
            result = int(np.where(dist==partn)[0][0])
            j+=1

        indices[i] = result

    return indices


def norm_onto_2d(input, target):
    """
    normalise one 2D array of values onto the other, by axis
    """

    if not ( len(input.shape) == 2 and len(target.shape) == 2):
        raise ValueError("both arrays must be 2D")

    target__ = np.copy(target)
    result = np.copy(input)

    for i in range(target.shape[1]):
        target__[:,i] = target__[:,i]-np.min(target[:,i])

        result[:,i] = np.copy(input[:,i])
        result[:,i] = result[:,i]-np.min(result[:,i])   

        result[:,i] = result[:,i]/np.max(result[:,i])
        result[:,i] = result[:,i]*np.max(target__[:,i])
        result[:,i] = result[:,i]+np.min(target[:,i])

    return result

def smartcast(data, target_dtype):
    """
    convert data to target_dtype
    handling safe casts and rounding for int->float
    """
    if not (np.issubdtype(target_dtype, np.number) and np.issubdtype(data.dtype, np.number)):
        raise ValueError("Both dtypes must be numeric")

    if target_dtype == data.dtype:
        data_ = data
    
    if np.issubdtype(data.dtype, np.integer) and np.issubdtype(target_dtype, np.integer):
        data_ = data.astype(target_dtype)

    elif np.issubdtype(data.dtype, np.floating) and np.issubdtype(target_dtype, np.floating):
        data_ = data.astype(target_dtype)

    elif np.issubdtype(data.dtype, np.floating) and np.issubdtype(target_dtype, np.integer):
        data_ = np.rint(data)
        data_ = data.astype(target_dtype)

    elif np.issubdtype(data.dtype, np.integer) and np.issubdtype(target_dtype, np.floating):
        data_ = data.astype(target_dtype)     
    else:
        "unexpected combination of dtypes - perhaps one or both are complex"

    if not np.allclose(data, data_, atol=1):
        raise ValueError("some values differ by >= 1.0 after cast")

    
    return data_
