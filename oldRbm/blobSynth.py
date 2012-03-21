#! /usr/bin/env ipython -pylab

from numpy import *
import pylab as p
from time import sleep



def gaussian(xx, yy, sigma = 1):
    '''Gaussian pdf in two dimensions (not normalized)'''

    rr2 = xx**2 + yy**2
    #print 'rr2 is', rr2
    #print exp(-rr2 / (2. * sigma**2))
    return exp(-rr2 / (2. * sigma**2))



def spritz(img, xx, yy, radius):
    for ii in range(-5, 6):
        for jj in range(-5, 6):
            #print 'ii, jj', ii, jj
            img[xx+ii, yy+jj] += gaussian(ii, jj, radius)



def generateBlobData(sizeX, sizeY):
    '''Generate some interesting data'''

    dat = zeros((sizeX, sizeY))

    xx = random.randint(5, sizeX-5)
    yy = random.randint(5, sizeY-5)
    rr = random.uniform(.5, 4.5)

    spritz(dat, xx, yy, rr)

    return dat, (xx, yy, rr)



def randomBlobs(N, sizeX = 20, sizeY = 20):
    for ii in xrange(N):
        yield generateBlobData(sizeX, sizeY)



def main():
    p.figure(1)

    for dat, params in randomBlobs(20, 20, 20):
        #p.imshow(dat, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        print 'params are', params
        p.imshow(dat, cmap='gray', interpolation='nearest')
        p.show()
        sleep(.1)



if __name__ == '__main__':
    main()
