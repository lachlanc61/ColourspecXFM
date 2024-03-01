#!/bin/bash

SIFDIR=$SIFDIR
IMAGENAME='xfmkit_latest'

APPDIR="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE}")")")"

cd $APPDIR/containers

apptainer build $SIFDIR/${IMAGENAME}.sif $APPDIR/containers/apptainer.def

#test
apptainer exec --containall --bind /tmp,$APPDIR/data:/data  $SIFDIR/xfmkit_latest.sif pytest /app

