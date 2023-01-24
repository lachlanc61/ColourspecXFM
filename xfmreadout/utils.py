import time
import sys
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.stats import norm

ANADIR="autoanalysis"

def getcfgs(f1, f2):
    """
    merges two dicts from filenames
        NB: watch duplicates, f2 will override
    """    
    dict1 = readcfg(f1)
    dict2 = readcfg(f2)

    return {**dict1, **dict2}


def readcfg(filename):
        dir = os.path.realpath(__file__) #_file = current file (ie. utils.py)
        dir=os.path.dirname(dir) 
        dir=os.path.dirname(dir)        #second call to get out of src/..

        yamlfile=os.path.join(dir,filename)

        with open(yamlfile, "r") as f:
            return yaml.safe_load(f)


def readargs():
    #get the arguments from command line
    parsed = argparse.ArgumentParser()

    parsed.add_argument("-c", "--usrconfig", help="User config file (.yaml)", type=os.path.abspath)
    parsed.add_argument("-i", "--infile", help="Input file (.GeoPIXE)", type=os.path.abspath)
    parsed.add_argument("-o", "--outdir", help="Output path", type=os.path.abspath)
    parsed.add_argument("-s", "--submap", action='store_true', help="Export submap (.GeoPIXE)")
    parsed.add_argument("-p", "--parse", action='store_true', help="Only export submap")
    parsed.add_argument("-f", "--force", action='store_true', help="Force recalculation of all pixels/classes")
    parsed.add_argument('-x', "--xcoords", nargs='+', type=int, help="X coordinates for submap as: xstart xend")
    parsed.add_argument('-y', "--ycoords", nargs='+', type=int, help="Y coordinates for submap as: ystart yend")
    parsed.add_argument('-ch', "--chunksize", nargs='+', type=int, help="Chunk size to load (in Mb)")

    return parsed.parse_args()


def initcfg(args, pkgconfig, usrconfig):
    #if the user config was given as an arg, use it
    if args.usrconfig is not None:
        usrconfig = args.usrconfig
    #otherwise just use the default 
    else:
        usrconfig = usrconfig
    
    #parse the config files 
    rawconfig=getcfgs(pkgconfig, usrconfig) 

    #create a working copy
    config=deepcopy(rawconfig)

    #modify working config based on args
    if args.infile is not None:
        config['infile'] = args.infile

    if args.outdir is not None:
        config['outdir'] = args.outdir

    if args.submap:
        config['WRITESUBMAP'] = True

    if args.parse:
        config['PARSEMAP'] = True
    else:
        print("EXPORTING SUBMAPS ONLY")

    if args.force:
        config['FORCEPARSE'] = True
        config['FORCERED'] = True
        config['FORCEKMEANS'] = True

    if args.chunksize is not None:
        config['chunksize'] = args.chunksize

    if args.xcoords is not None:
        config['submap_x'][0]=args.coords[0]
        config['submap_x'][1]=args.coords[1]

        if not config['WRITESUBMAP']:
            print("WARNING: submap coordinates set but submap flag False")

    if args.ycoords is not None:
        config['submap_y'][0]=args.coords[2]
        config['submap_y'][1]=args.coords[3]

        if not config['WRITESUBMAP']:
            print("WARNING: submap coordinates set but submap flag False")

    if not config['PARSEMAP']:
        config['DOCOLOURS']=False
        config['DOCLUST']=False
        config['DOBG']=False

    if config['WRITESUBMAP']:
        if config['submap_x'][1] == 0:
            config['submap_x'][1]=int(99999)
        if config['submap_y'][1] == 0:
            config['submap_y'][1]=int(99999)

        if (config['submap_x'][0] >= config['submap_x'][1]):
            raise ValueError("FATAL: x2 nonzero but smaller than x1")
        if (config['submap_y'][0] >= config['submap_y'][1]):
            raise ValueError("FATAL: y2 nonzero but smaller than y1")
    return config, rawconfig

class DirectoryStructure:
    def __init__(self, config):
        """
        Assign input and output directories from the config
            or location of script if relative
        """
        self.script = os.path.realpath(__file__) #_file = current script
        self.spath=os.path.dirname(self.script) 
        self.spath=os.path.dirname(self.spath)
        
        #check if paths are absolute or relative based on leading /
        if config['infile'][0].startswith('/'):
            self.fi=config['infile'][0]
        else:
            self.fi = os.path.join(self.spath,config['infile'][0])

        #assign output:
        #   to input if output blank
        #   otherwise as assigned
        if config['outdir'][0] == "" or config['outdir'][0] == None:
            self.odir=os.path.join(os.path.dirname(self.fi),config['ANADIR'])
        elif config['outdir'][0].startswith('/'):   #relative vs absolute
            self.odir=config['outdir'][0]
        else:
            self.odir=os.path.join(self.spath,config['outdir'][0])
    
        if self.odir.endswith('/'):
            self.odir = os.path.dirname(self.odir)

        #extract terminal directory for output
        outbase=os.path.basename(self.odir)

        #if terminal directory has the correct name(s), continue
        #   otherwise, append it
        if outbase == config['ANADIR']:      
            pass
        else:
            self.odir=os.path.join(self.odir,config['ANADIR'])

        #assign and create analysis subdirs, if needed
        self.transforms=os.path.join(self.odir,config['TRANSDIR'])
        self.plots=os.path.join(self.odir,config['PLOTDIR'])
        self.exports=os.path.join(self.odir,config['EXPORTDIR'])

        #extract name of input file
        self.fname = os.path.splitext(os.path.basename(self.fi))[0]

        #setup submap export location and extension
        if config['WRITESUBMAP']:
            self.subname=self.fname+config['convext']
            self.fsub = os.path.join(self.exports,self.subname+config['FTYPE'])

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
        
        for dir in [ self.transforms, self.plots, self.exports ]:
            if not os.path.isdir(dir):
                os.mkdir(dir)

        return self

    def check(self, config):
        """
        run some basic sanity checks
            eg. correct filetype
        """
        #check filetype is recognised - currently hardcoded
        if not config['FTYPE'] == ".GeoPIXE":
            raise ValueError(f"Filetype {config['FTYPE']} not recognised")
    
        for dir in [ self.odir, self.transforms, self.plots, self.exports ]:
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

def initfiles(config):

    dirs = DirectoryStructure(config)
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
    xout=np.zeros(nchannels,dtype=int)
    yout=np.zeros(nchannels, dtype=int)

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


def pxinsubmap(config, xcoord, ycoord):
    if (xcoord >= config['submap_x'][0] and xcoord < config['submap_x'][1] and
            ycoord >= config['submap_y'][0] and ycoord < config['submap_y'][1]
    ):
        return True
    else:
        return False