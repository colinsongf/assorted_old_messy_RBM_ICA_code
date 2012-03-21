#! /usr/bin/env ipython -pylab

from numpy import *
import pylab as p
from time import sleep



def generateSimpleLRData(size):
    '''Generate some simple correlated binary data

    Each data vector is randomly selected from the following two classes:
    [0, ..., 0, 1, ..., 1]
    [1, ..., 1, 0, ..., 0]
    where the length of each vector is 2*size
    '''

    if random.randint(2) == 0:
        theClass = 0
        dat = hstack((zeros((size,)), ones((size,))))
    else:
        theClass = 1
        dat = hstack((ones((size,)), zeros((size,))))

    return dat, theClass



def randomSimpleLR(N, size = 2):
    if N < 0:
        while True:
            yield generateSimpleLRData(size)
    else:
        for ii in xrange(N):
            yield generateSimpleLRData(size)



def main():
    p.figure(1)

    for dat, params in randomSimpleLR(20, 2):
        #p.imshow(dat, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        print 'params are', params
        p.imshow(array([dat]), cmap='gray', interpolation='nearest')
        p.show()
        sleep(1)



if __name__ == '__main__':
    main()
