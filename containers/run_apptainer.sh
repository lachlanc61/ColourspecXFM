
SIFDIR=$SIFDIR
IMAGENAME='xfmreadout_latest'
APPDIR="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE}")")")"
DATADIR="$APPDIR/data"

binds="/tmp,$DATADIR:/data,$APPDIR:/repo"

apptainer run --containall --bind $binds  $SIFDIR/$IMAGENAME.sif python -f /data/example_datafile.GeoPIXE -m


apptainer exec 