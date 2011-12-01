#! /usr/bin/env ipython --pylab

from numpy import *
import pylab as pl
from time import sleep
import pdb


def main():
    pts = array([[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 1],
                 [1, 1, 1],
                 ])
    
    R = array([
        [1, 1, 0],
        [0, 0, 1],
        ], dtype=float)

    for ii in range(R.shape[0]):
        R[ii,:] /= linalg.norm(R[ii,:])

    plotXY = dot(pts, R.T)

    print plotXY

    #curSubplot = 1
    #
    #ax = pl.subplot(3,1,curSubplot)
    #curSubplot += 1
    ##pl.imshow(self._h.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    #pl.xticks([])
    #pl.yticks([])
    #pl.axvline(.5, color=[1,.4,.4,1], linewidth=2)
    #
    #pl.show()
    #raw_input('Enter to exit.')





if __name__ == '__main__':
    main()
