#!/bin/bash

SIFDIR=$SIFDIR
IMAGENAME='xfmreadout_latest'

apptainer build $SIFDIR/${IMAGENAME}.sif apptainer.def


#to run
#apptainer exec --containall --bind /tmp,$PWD/data:/data  $SIFDIR/xfmreadout_latest.sif xfmread-raw -f /data/example_datafile.GeoPIXE -m
