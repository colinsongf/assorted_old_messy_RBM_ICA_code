#! /usr/bin/env python

from numpy import *
from numdifftools import Gradient
import pdb

from ica.tica import TICA
from ica.rica import RICA
from util.dataLoaders import loadFromPklGz



def fun(x):
    A = array([[1, 2], [3, 4]])
    b = array([3, 5])

    cost = .5 * dot(x.T, dot(A, x)) + dot(b, x)

    grad = .5 * dot(A + A.T, x) + b

    return cost, grad



def testRica():
    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    data = data[:9, :500]   # take just a small slice of data
    
    #pdb.set_trace()

    random.seed(0)
    rica = RICA(imgShape = (3, 3),
                nFeatures = 10,
                lambd = .05,
                epsilon = 1e-5)
    normData = True
    if normData:
        # Project each patch to the unit ball
        patchNorms = sqrt(sum(data**2, 0) + (1e-8))
        data = data / patchNorms
    
    # Initialize weights WW
    WW = random.randn(10, 3*3)
    WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
    #loadedWW = loadtxt('../octave-randTheta')
    #WW = loadedWW
    WW = WW.flatten()

    cost, sGrad = rica.cost(WW, data)
    print 'Current cost is:', cost
    print 'Gradient (symbolic) is:', sGrad

    nGradFn = Gradient(lambda w: rica.cost(w, data)[0])
    nGrad = nGradFn(WW)
    print 'Gradient (finite diff) is:', nGrad
    print 'diff is', sGrad - nGrad

    pdb.set_trace()



def testTica():
    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    data = data[:25, :500]   # take just a small slice of data
    
    #pdb.set_trace()

    random.seed(0)
    tica = TICA(imgShape = (5, 5),
                hiddenLayerShape = (4, 5),
                shrink = 0,
                lambd = .05,
                epsilon = 1e-5)
    normData = True
    if normData:
        # Project each patch to the unit ball
        patchNorms = sqrt(sum(data**2, 0) + (1e-8))
        data = data / patchNorms
    
    # Initialize weights WW
    WW = random.randn(4*5, 5*5)
    WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
    #loadedWW = loadtxt('../octave-randTheta')
    #WW = loadedWW
    WW = WW.flatten()

    cost, sGrad = tica.cost(WW, data)
    print 'Current cost is:', cost
    print 'Gradient (symbolic) is:', sGrad

    nGradFn = Gradient(lambda w: tica.cost(w, data)[0])
    nGrad = nGradFn(WW)
    print 'Gradient (finite diff) is:', nGrad
    print 'diff is', sGrad - nGrad

    pdb.set_trace()



def main():
    x0 = array([5, 7])
    print 'fun(x0)  =', fun(x0)
    dfun = Gradient(lambda x: fun(x)[0])
    print 'dfun(x0) =', dfun(x0)
    print

    data = random.randn(9, 16)

    # Initialize weights WW
    W0 = random.randn(16, 9)
        
    tica = TICA(imgShape = (3, 3),
                hiddenLayerShape = (4, 4),
                neighborhoodSize = 1,
                lambd = .1,
                epsilon = 1e-5)

    randIdx = random.randint(0, prod(W0.shape), 10)

    print 'tica.cost(W0, data) = ', tica.cost(W0, data)
    dcost = Gradient(lambda w : tica.cost(w.reshape(16,9), data)[0])
    print 'd(tica.cost) = ... ', dcost(W0.flatten())[randIdx]

    pdb.set_trace()


if __name__ == '__main__':
    #testRica()
    testTica()
    #main()
