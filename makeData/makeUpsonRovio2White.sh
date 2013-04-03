#! /bin/bash

dirload='../data/upson_rovio_2'
dirsave='../data/upson_rovio_2/white'

for size in 02 03 04 06 08 10 15 20 25 28 30 40; do
#for size in 02 04 10; do
    for npoints in 50 50000; do
        for colors in 1 3; do
            # start two at once
            for tt in train test; do
                set -x
                ./saveWhite.py loadUpsonData \
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
