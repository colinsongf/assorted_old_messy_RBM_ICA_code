#! /usr/bin/env ipython -pylab

#! /usr/bin/env python

import sys
from time import sleep

from numpy import *
from numpy.random import *
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90

def main():
    for arg in sys.argv:
        print 'arg is', arg

    p.figure(1)
    p.plot(xrange(10),rand(10), 'b+')
    p.figure(2)
    p.plot(xrange(10),rand(10), 'r+')

    p.figure(3)
    for ii in range(10):
        p.imshow(rand(10,10), cmap='gray', interpolation='nearest')
        p.show()
        sleep(.5)



main()
