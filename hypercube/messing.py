#! /usr/bin/env python
#! /usr/bin/env ipython --pylab
#! /usr/bin/env ipython --pylab=osx
#! /usr/bin/env ipython --pylab=wx

from numpy import *
import matplotlib

matplotlib.use("Agg")

import pylab as pl
from time import sleep
import pdb

from helper import *



def main():
    dim = 16
    corners, (edges0, edges1) = hypercubeCornersEdges(dim)

    baseRot = eye(dim)
    w = array([[1 for ii in range(dim)]])
    w = random.random(dim)
    baseRot[0,:] = w
    print 'baseRot is\n', baseRot
    gramSchmidt(baseRot)
    print 'baseRot is\n', baseRot
    if linalg.det(baseRot) < 0:
        baseRot[-1,:] = -baseRot[-1,:]

    baseRot = baseRot.T
    #baseRot = eye(dim)
    deltaRot = rotationMatrix(dim, None, .01, skipFirst = True)

    rot = copy(baseRot)
    for iteration in xrange(2000):
        rot = dot(rot, deltaRot)
        cornersRot  = dot(corners, rot)
        edges0Rot   = dot(edges0, rot)
        edges1Rot   = dot(edges1, rot)
        #pl.figure(0)
        pl.cla()
        #print 'corners:\n', corners
        #print 'cornersRot:\n', cornersRot
        pl.plot(cornersRot[:,0], cornersRot[:,1], 'ro')

        ax = pl.gca()
        for ii in range(edges0Rot.shape[0]):
            line = pl.Line2D([edges0Rot[ii,0], edges1Rot[ii,0]],
                             [edges0Rot[ii,1], edges1Rot[ii,1]],
                             linewidth = 1)
            ax.add_line(line)

        if iteration == 0:
            pl.axis('equal')
            pl.draw()
            axlim = looser(pl.axis(), .3)
        pl.axis(axlim)
        pl.draw()
        pl.savefig('rot_%02d_%05d.png' % (dim, iteration))
        sleep(.001)

    raw_input('Enter to exit.')



if __name__ == '__main__':
    main()
