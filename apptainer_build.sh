#!/bin/bash

SIFDIR='/home/lachlan/SIFSTORE/'
IMAGENAME='xfmreadout_latest'

apptainer build $SIFDIR/${IMAGENAME}.sif apptainer.def