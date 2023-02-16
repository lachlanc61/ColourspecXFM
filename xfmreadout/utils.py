import time
import sys
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.stats import norm

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
        
        #check if paths are absolute or relative based on leading /
        if args.input_file.startswith('/'):
            self.fi=args.input_file
        else:
            self.fi = os.path.join(self.spath,args.input_file)

        #assign output:
        #   to input if output blank
        #   otherwise as assigned
        if args.output_directory == "" or args.output_directory == None:
            self.odir=os.path.join(os.path.dirname(self.fi),config['ANADIR'])
        elif args.output_directory.startswith('/'):   #relative vs absolute
            self.odir=args.output_directory
        else:
            self.odir=os.path.join(self.spath,args.output_directory)
    
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
        if args.write_modified:
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
            eg. check for correct filetype
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


def pxinsubmap(xin, yin, xread, yread):
    if (xread >= xin[0] and xread < xin[1] and
            yread >= yin[0] and yread < yin[1]
    ):
        return True
    else:
        return False