
apptainer run --containall --bind /tmp,$PWD/data:/data  $SIFDIR/xfmreadout_latest.sif -f /data/example_datafile.GeoPIXE -m


