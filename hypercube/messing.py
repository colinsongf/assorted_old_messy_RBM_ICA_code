#! /usr/bin/env ipython --pylab
#! /usr/bin/env ipython --pylab=osx
#! /usr/bin/env ipython --pylab=wx

from numpy import *
import matplotlib
import pylab as pl
from time import sleep
import pdb

from helper import *



def main():
    dim = 3
    corners, (edges0, edges1) = hypercubeCornersEdges(dim)

    rot = eye(dim)
    w = array([[1, 1, 1]])
    rot[0,:] = w
    print 'rot is\n', rot
    gramSchmidt(rot)
    print 'rot is\n', rot
    if linalg.det(rot) < 0:
        rot[-1,:] = -rot[-1,:]

    rot = rot.T
    #rot = eye(dim)

    cornersRot = dot(corners, rot)
    edges0Rot   = dot(edges0, rot)
    edges1Rot   = dot(edges1, rot)
    pl.figure()
    print 'corners:\n', corners
    print 'cornersRot:\n', cornersRot
    pl.plot(cornersRot[:,0], cornersRot[:,1], 'ro')

    ax = pl.gca()
    for ii in range(edges0Rot.shape[0]):
        line = Line2D([edges0Rot[ii,0], edges1Rot[ii,0]],
                      [edges0Rot[ii,1], edges1Rot[ii,1]])
        ax.add_line(line)

    pl.axis(looser(pl.axis()))

    #plt.ion()
    #pl.show()
    assert 0

    raw_input('Enter to exit.')



if __name__ == '__main__':
    main()
