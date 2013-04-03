#! /bin/bash

dirload='../data/atari'
dirsave='../data/atari/white'

for size in 02 03 04 06 10 15 20 25 28; do
#for size in 02; do
    for npoints in 50 50000; do
        for colors in 1 3; do
            # start four ~equal sized jobs at once
            for game in mspacman space_invaders; do
                for tt in train test; do
                    set -x
                    ./saveWhite.py loadAtariData \
                        $dirload/${game}_${tt}_${size}_${npoints}_${colors}c.pkl.gz \
                        $dirsave/${game}_${tt}_${size}_${npoints}_${colors}c.white.pkl.gz \
                        $dirsave/${game}_${tt}_${size}_${npoints}_${colors}c.whitenormed.pkl.gz  \
                        $dirsave/${game}_${tt}_${size}_${npoints}_${colors}c.whitener.pkl.gz &
                    set +x
                done
            done
            pids=$(jobs -p)
            echo "`date` `date +%s` Waiting jobs to finish: $pids"
            wait $pids
            echo "`date` `date +%s` pids finished"
        done
    done
done
