#!/bin/bash

SIFDIR=$SIFDIR
IMAGENAME='xfmreadout_latest'

APPDIR="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE}")")")"

apptainer build $SIFDIR/${IMAGENAME}.sif $APPDIR/containers/apptainer.def

#test
apptainer exec --containall --bind /tmp,$APPDIR/data:/data  $SIFDIR/xfmreadout_latest.sif pytest /app

