#! /usr/bin/env ipython -pylab

'''
Research code

Jason Yosinski
'''

from numpy import *
import pylab as p
from time import sleep, time, clock
import pdb
import timeit

from RBM import RBM
from blobSynth import randomBlobs

# Todo:
# - learning!



def plot(rbm, result, titleStr, centerStr):
    p.clf()

    rbm.plot(skipW = True, skipV = True)
    p.title(titleStr)

    ax = p.subplot(3,1,2)
    #p.imshow(result, cmap='gray', interpolation='nearest', vmin = 0, vmax = 1)
    p.imshow(result, cmap='gray', interpolation='nearest')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    p.title(centerStr)

    p.subplot(3,1,3)
    p.plot(rbm.reconErrorNorms)
    p.title('Reconstruction error (mean last 10 = %.2f)' % mean(rbm.reconErrorNorms[-10:]))
    p.show()



def main():
    random.seed(0)
    size = 40
    rbm = RBM(size*size, 1000)

    slp = 2
    #for blob, params in randomBlobs(10, size, size):
    #    rbm.v = blob.flatten()
    #    plot(rbm, blob)
    #    sleep(slp)
    #
    #    result = reshape(rbm.v, (size, size))
    #    plot(rbm, result)
    #    sleep(slp)
    #
    #    rbm.v2h()
    #    plot(rbm, blob)
    #    sleep(slp)
    #
    #    rbm.h2v()
    #    result = reshape(rbm.v, (size, size))
    #    plot(rbm, result)
    #    sleep(slp)

    every = 2000

    #m = log(.01/.1) / log(10000)
    NN = 20001
    bb = 1. / NN * log(.001 / 1.0)
    elapsed = array([])
    for ii in range(NN):
        if ii % 100 == 0:
            blob, params = randomBlobs(10, size, size).next()
        
        #print params

        #epsilon = .1 * (ii+1) ** m
        #epsilon = .3 * exp(bb * ii)
        epsilon = min(.1, 1.0 * exp(bb * ii))

        #time0 = time()
        rbm.learn1(blob.flatten(), epsilon = epsilon, activationH2V = 'gaussianReal', param = 1)
        #elapsed = hstack((elapsed, time() - time0))

        if ii % every == 0:
            print '%d: epsilon is %f' % (ii, epsilon),
            rbm.v = blob.flatten()
            result = reshape(rbm.v, (size, size))
            plot(rbm, result, 'Iteration %d' % ii, 'Data')
            p.show()
            sleep(.1) if ii == 0 else sleep(.1)

            rbm.v2h()
            rbm.h2v(activation = 'gaussianReal', param = 0)
            
            result = reshape(rbm.v, (size, size))
            plot(rbm, result, 'Iteration %d' % ii, 'Reconstruction')
            p.show()
            sleep(.5) if ii == 0 else sleep(.5)

            print 'mean of last 50 errors is', mean(rbm.reconErrorNorms[-50:])

            #print 'average elapsed is:', mean(elapsed)
            #elapsed = array([])



if __name__ == '__main__':
    main()
