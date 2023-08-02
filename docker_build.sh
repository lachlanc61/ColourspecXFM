#!/bin/bash

IMAGENAME='xfmkit:0.1'

docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
     -t "$IMAGENAME" .

#to run example data on host
#docker run -it -v /home/lachlan/CODEBASE/ReadoutXFM/data/TEMP:/data xfmkit:0.1 xfmread-raw -f /data/example_datafile.GeoPIXE -m