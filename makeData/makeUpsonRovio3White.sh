#! /bin/bash

dirload='../data/upson_rovio_3'
dirsave='../data/upson_rovio_3/white'

for size in 02 03 04 06 08 10 15 20 25 28 30 40; do
#for size in 02 04; do
    for npoints in 50 50000; do
    #for npoints in 50; do
        for colors in 1 3; do
            # start three at once
            #for tt in train test explore; do
            #for tt in train test; do
            for tt in explore; do
                set -x
                ./saveWhite.py loadUpsonData3 \
                    $dirload/${tt}_${size}_${npoints}_${colors}c.pkl.gz \
                    $dirsave/${tt}_${size}_${npoints}_${colors}c.white.pkl.gz \
                    $dirsave/${tt}_${size}_${npoints}_${colors}c.whitenormed.pkl.gz  \
                    $dirsave/${tt}_${size}_${npoints}_${colors}c.whitener.pkl.gz &
                set +x
            done
            pids=$(jobs -p)
            echo "`date` `date +%s` Waiting jobs to finish: $pids"
            wait $pids
            echo "`date` `date +%s` pids finished"
        done
    done
done
