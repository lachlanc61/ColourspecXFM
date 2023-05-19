#!/bin/bash

SIFDIR=$SIFDIR
IMAGENAME='xfmreadout_latest'

apptainer build $SIFDIR/${IMAGENAME}.sif apptainer.def

#test
apptainer exec --containall --bind /tmp,$PWD/../data:/data  $SIFDIR/xfmreadout_latest.sif pytest /app

#to run
#apptainer run --containall --bind /tmp,$PWD/data:/data  $SIFDIR/xfmreadout_latest.sif -f /data/example_datafile.GeoPIXE -m

