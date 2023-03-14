
import csv
import re
import os
import sys
import re
import argparse
import numpy as np


OUT_NAME="diag.tmp"

testpath="/home/lachlan/CODEBASE/ReadoutXFM/data/diagnostics_map1_evts.log"
testpath="/mnt/d/DATA/XFMDATA/2023/Lachlan/REFERENCE/uMatter/Mo_vac_230313/50-200_TC1p0/diagnostics.log"
testname="um"

NDET=2

def checkargs(args):
    if args.input_file == None:   
        raise ValueError("No input file specified")

    return args 

def getargs(args_in):
    """
    parse command line arguments
    """

    argparser = argparse.ArgumentParser(
        description="Utility to parse IXRF log file for raw deadtime statistics"
    )

    #--------------------------
    #set up the expected args
    #--------------------------
    #inputs and outputs locations
    argparser.add_argument(
        "-f", "--input-file", 
        help="Specify a .log file to be read in", 
        type=os.path.abspath,
    )
    argparser.add_argument(
        "-s", "--split", 
        help="Split file into separate logs based on filenames in log",
        action='store_true',
    )

    args = argparser.parse_args(args_in)

    args = checkargs(args)

    return args


def dtfromdiag(filepath: str):
    """
    extracts per-pixel deadtime values from IXRF diagnostic file

    (IXRF uses nonstandard utf8-as-utf16 format with null bytes)
    
    IMPORTANT: requires diagnostic files to be converted via sed:
    """
        #   sed -i 's/\x0//g' diagnostics_map1_evts.log

    with open(filepath) as f:
        nlines_init=sum(1 for line in f)
        print(nlines_init)
        f.seek(0)

        rt=np.zeros((NDET, nlines_init), dtype=float)
        lt=np.zeros((NDET, nlines_init), dtype=float)
        tr=np.zeros((NDET, nlines_init), dtype=int)
        ev=np.zeros((NDET, nlines_init), dtype=int)
        ocr=np.zeros((NDET, nlines_init), dtype=float)
        icr=np.zeros((NDET, nlines_init), dtype=float)
        dt=np.zeros((NDET, nlines_init), dtype=float)

        print(rt.shape, rt[1], nlines_init)

        nlines=0
        npx=0
        nmaps=0
        cdet=0

        for line in csv.reader(f):
            if "Map Acquire start" in line[0]:
                print(f"Map started at line: {nlines}")
                nmaps += 1
            if "Deadtime realtime" in line[0]:
                rt[cdet, npx] = float(re.findall("[\d]+[\.]\d+", line[0])[0])      
                lt[cdet, npx] = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[1])[0])
                tr[cdet, npx ]= int(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[2])[0])
                ev[cdet, npx] = int(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[3])[0])
                ocr[cdet, npx ]= float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[4])[0])
                icr[cdet, npx] = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[5])[0])
                print(cdet, rt[cdet, npx], lt[cdet, npx], tr[cdet, npx], ev[cdet, npx], ocr[cdet, npx], icr[cdet, npx] )
            
            if "deadtime[0]" in line[0]:
                #print("DEADTIME0")
                if not cdet == 0: 
                    raise ValueError(f"Detector: expected 0, found {cdet} at line {nlines}, pixel {npx} ")
                dt[cdet, npx] = float(re.findall("[\d]+[\.]\d+", line[0])[0])
                cdet=1  #next detector

            if "deadtime[1]" in line[0]:
                #print("DEADTIME1")
                if not cdet == 1: 
                    raise ValueError(f"Detector: expected 0, found {cdet} at line {nlines}, pixel {npx} ")
                dt[cdet, npx] = float(re.findall("[\d]+[\.]\d+", line[0])[0])
                cdet=0  #first detector
                npx+=1  #next pixel

            if "Saving geoPIXE map file as" in line[0]:    #35 = character count for end of prestring
                if nmaps > 1:
                    print("WARNING: MULTIPLE MAPS present, resetting")
                    rt=np.zeros((NDET, nlines_init), dtype=float)
                    lt=np.zeros((NDET, nlines_init), dtype=float)
                    tr=np.zeros((NDET, nlines_init), dtype=int)
                    ev=np.zeros((NDET, nlines_init), dtype=int)
                    ocr=np.zeros((NDET, nlines_init), dtype=float)
                    icr=np.zeros((NDET, nlines_init), dtype=float)
                    dt=np.zeros((NDET, nlines_init), dtype=float)
                    npx=0
            nlines+=1    

        rt=rt[:, :npx]
        lt=lt[:, :npx]
        tr=tr[:, :npx]
        ev=ev[:, :npx]
        ocr=ocr[:, :npx]
        icr=icr[:, :npx]
        dt=dt[:, :npx]
        calc_dt=100*(1-ocr/icr)     #ICR/OCR
        #calc_dt=100*(rt-lt)/rt      #real/live

    print(f"last values: {rt[1, -1]} {lt[1, -1]} {tr[1, -1]} {ev[1, -1]} {ocr[1, -1]} {icr[1, -1]}")

    print(f"lines found: {nlines_init}, lines read: {nlines}, pixels: {npx}, stored: {len(rt[0])}")

    print(f"IXRF DT -- max: {round(np.max(dt),2)}, avg: {round(np.average(dt),2)}")
    print(f"calc DT -- max: {round(np.max(calc_dt),2)}, avg: {round(np.average(calc_dt),2)}")

    return rt, lt, tr, ev, icr, ocr, calc_dt

def splitlog(filepath: str):
    """
    Takes a diagnostic log and splits it into separate files per map

    outputs are named "diagnostics_[map_name].log"
        [map_name] is read from .GeoPIXE file in log

    if multiple maps in the log share the same name, only the most recent is split
    """

    out_path=os.path.dirname(os.path.abspath(filepath))
    tempf=os.path.join(out_path, OUT_NAME)

    with open(filepath) as f:
       
        line_idx = 0
        do_export=False

        if os.path.isfile(tempf):
            os.remove(tempf)
        
        tempfile=open(tempf, 'a')

        for line in csv.reader(f):
            if "FastMap::Init()" in line[0]:
                do_export=True
                start_idx=line_idx
                
            if do_export:
                line_as_str=','.join(line)
                line_as_str+='\n'
                tempfile.write(line_as_str)

            if "Saving geoPIXE map file as" in line[0]:
                
                name=re.findall(r"[a-zA-Z0-9\-\_]+.GeoPIXE",line[0])[0]
                name=os.path.splitext(name)[0]
                timestamp=re.findall(r"[0-9]+\:[0-9]+\:[0-9]+",line[0])[0]

                tempfile.close()
                newf=os.path.join(out_path, f"diagnostics_{name}.log")
                print(f"time: {timestamp}, lines: {start_idx} to {line_idx}, file: {name}.GeoPIXE")
                print(f"saving to: {newf}")

                if os.path.isfile(newf):
                    print(f"WARNING: previous file overwritten for name {name}")

                os.rename(tempf, newf)

                tempfile=open(tempf, 'a')
                do_export=False

            line_idx += 1

        tempfile.close()

        if os.path.isfile(tempf):
            os.remove(tempf)
    return


def main(args_in):
    #get command line arguments
    args = getargs(args_in)

    #check if filepath is absolute based on leading /
    if args.input_file.startswith('/'):
        fi=args.input_file
    else:
        fi = os.path.join(os.getcwd(),args.input_file)

    if args.split:
        splitlog(fi)
        return None, None, None, None, None, None, None
    else:
        rt, lt, tr, ev, icr, ocr, calc_dt = dtfromdiag(fi)

    return rt, lt, tr, ev, icr, ocr, calc_dt


if __name__ == "__main__":
    main(sys.argv[1:])      #NB: exclude 0 == script name

    sys.exit()

