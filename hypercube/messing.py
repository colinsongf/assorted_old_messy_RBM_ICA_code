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

    rot = eye(3)
    w = array([[1, 1, 1]])
    rot[0,:] = w
    gramSchmidt(rot)
    print 'rot is', rot

    raw_input('Enter to exit.')



if __name__ == '__main__':
    main()
