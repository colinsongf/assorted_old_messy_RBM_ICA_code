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
from simpleSynth import randomSimpleLR



def pause():
    raw_input('')



def plot(rbm, titleStr, centerStr):
    p.clf()

    #rbm.plot(skipW = True, skipV = True)
    rbm.plot(nSubplots = 4)
    p.subplot(4,1,1)
    p.title(titleStr + ' -- ' + centerStr)

    #ax = p.subplot(3,1,2)
    #p.imshow(result, cmap='gray', interpolation='nearest', vmin = 0, vmax = 1)
    #p.imshow(array([result]), cmap='gray', interpolation='nearest')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    #p.title(titleStr + centerStr)
    
    p.subplot(4,1,4)
    p.plot(rbm.reconErrorNorms)
    p.title('Reconstruction error (mean last 10 = %.2f)' % mean(rbm.reconErrorNorms[-10:]))
    p.show()



def main():
    random.seed(0)
    size = 2
    rbm = RBM(size*2, 1)

    every = 40000
    #every = 100

    #m = log(.01/.1) / log(10000)
    NN = 40001
    bb = 1. / NN * log(.001 / 1.0)
    elapsed = array([])
    dataGenerator = randomSimpleLR(-1, size)
    for ii in range(NN):
        #if ii % 100 == 0:
        data, params = dataGenerator.next()
        
        #print params

        #epsilon = .1 * (ii+1) ** m
        #epsilon = .3 * exp(bb * ii)
        #epsilon = min(.1, 1.0 * exp(bb * ii))
        epsilon = .1

        #time0 = time()
        #rbm.learn1(datablob.flatten(), epsilon = epsilon, activationH2V = 'gaussianReal', param = 1)
        #elapsed = hstack((elapsed, time() - time0))

        if ii % every == 0:
            print 'Iteration %d' % ii
            #print '%d: epsilon is %f' % (ii, epsilon),
            rbm.v = data
            plot(rbm, 'Iteration %d' % ii, '0. Data')
            print 'Visible set to:'
            print rbm.v.T
            p.show()
            pause()
            #sleep(1) if ii == 0 else sleep(.1)

            rbm.v2h()
            plot(rbm, 'Iteration %d' % ii, '1. To hidden')
            print 'W * visible (then sampled) ='
            print dot(rbm._W, rbm._v).T[:,1:]
            print rbm.h.T
            p.show()
            pause()

            rbm.h2v(activation = 'logisticBinary')
            plot(rbm, 'Iteration %d' % ii, '2. To visible')
            print 'W.T * hidden (then sampled) ='
            print dot(rbm._W.T, rbm._h).T[:,1:]
            print rbm.v.T
            p.show()
            pause()

            print
            
            #sleep(.5) if ii == 0 else sleep(.5)

            #print 'mean of last 50 errors is', mean(rbm.reconErrorNorms[-50:])

            #print 'average elapsed is:', mean(elapsed)
            #elapsed = array([])

        rbm.learn1(data, epsilon = epsilon, activationH2V = 'logisticBinary')
            



if __name__ == '__main__':
    main()
