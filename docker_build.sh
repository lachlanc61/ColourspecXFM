#!/bin/bash

IMAGENAME='xfmreadout:0.1'

docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
     -t "$IMAGENAME" .
