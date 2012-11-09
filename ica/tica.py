#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
from numpy import *

from util.ResultsManager import resman
from util.dataLoaders import loadFromPklGz, saveToFile
from rica import RICA, l2RowScaled, l2RowScaledGrad

try:
    from numdifftools import Gradient
except ImportError:
    print 'Could not import numdifftools. Probably fine.'



def neighborMatrix(hiddenLayerShape, neighborhoodSize, shrink = 0):
    '''Generate the neighbor matrix H, a 4D tensor where for the original tensor:
    
        H_i,j,k,l = 1 if the pooled unit at i,j is connected to the hidden unit at k,l

    Note: the tensor is returned flattened to 2D.

    hiddenLayerShape: size of the hidden layer, like (20,30)
    neighborhoodSize: number of pixels to grow the neighborhood
        by. e.g. ns=0 -> no pooling, ns=1 -> 3x3 neighborhoods.
    shrink: shrink each edge of the pooling units by this much vs. the
        hidden layer size. Just added to illustrate that the number of
        pooling neurons need not equal the number of hidden neurons.
    '''
    
    if shrink < 0:
        raise Exception('shrink parameter must be >= 0')
    if shrink*2 >= min(hiddenLayerShape):
        raise Exception('shrink parameter must be < min(hiddenLayerShape)/2')
        
    pooledLayerShape = (hiddenLayerShape[0]-2*shrink, hiddenLayerShape[1]-2*shrink)
    nHidden = prod(hiddenLayerShape)
    nPooled = prod(pooledLayerShape)

    # Create the 4D neighborhood tensor.
    # ret_i,j,k,l = 1 if the pooled unit at i,j is connected to the hidden unit at k,l
    ret = zeros(pooledLayerShape + hiddenLayerShape)

    rangeNeighborII = range(-neighborhoodSize, neighborhoodSize + 1)
    rangeNeighborJJ = range(-neighborhoodSize, neighborhoodSize + 1)

    for ii in range(pooledLayerShape[0]):
        for jj in range(pooledLayerShape[1]):
            for nnii in rangeNeighborII:
                for nnjj in rangeNeighborJJ:
                    ret[ii, jj, (ii+shrink+nnii) % hiddenLayerShape[0], (jj+shrink+nnjj) % hiddenLayerShape[1]] = 1

    return reshape(ret, (nPooled, nHidden))



def fullNeighborMatrix(hiddenLayerShape, neighborhoodSize):
    return neighborMatrix(hiddenLayerShape, neighborMatrix, shrink = 0)



class TICA(RICA):
    '''See RICA for constructor arguments.'''

    def __init__(self, imgShape, lambd = .005, hiddenLayerShape = (10,10), neighborhoodSize = 1,
                 shrink = 0, epsilon = 1e-5, saveDir = '',
                 float32 = False):
        self.hiddenLayerShape = hiddenLayerShape
        
        super(TICA, self).__init__(imgShape = imgShape,
                                   lambd = lambd,
                                   nFeatures = prod(self.hiddenLayerShape),
                                   epsilon = epsilon,
                                   float32 = float32,
                                   saveDir = saveDir)

        self.neighborhoodSize = neighborhoodSize
        self.shrink = shrink
        self.nPooled = (self.hiddenLayerShape[0] - self.shrink*2) * (self.hiddenLayerShape[1] - self.shrink*2)
        self.HH = neighborMatrix(self.hiddenLayerShape, self.neighborhoodSize, shrink = self.shrink)
        if self.float32:
            self.HH = array(self.HH, dtype='float32')


    def cost(self, WW, data):
        '''Main method of TICA that differs from RICA.'''

        nInputDim = data.shape[0]
        nDatapoints = data.shape[1]
        if self.nInputDim != nInputDim:
            raise Exception('Expected shape %s = %d dimensional input, but got %d' % (repr(self.imgShape), self.nInputDim, nInputDim))

        if self.float32:
            WW = array(WW, dtype='float32')
        
        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nFeatures, nInputDim)
        WWold = WW
        WW = l2RowScaled(WW)

        # % Forward Prop
        # h = W*x;
        # r = W'*h;
        hidden = dot(WW, data)
        reconstruction = dot(WW.T, hidden)
        
        #pdb.set_trace()
        #print 'TICA HERE'

        # % Reconstruction Loss and Back Prop
        # diff = (r - x);
        # reconstruction_cost = 0.5 * sum(sum(diff.^2));
        # outderv = diff;
        reconDiff = reconstruction - data
        reconstructionCost = sum(reconDiff ** 2)

        # L2 Pooling / Sparsity cost
        absPooledActivations = sqrt(self.epsilon + dot(self.HH, hidden ** 2))
        poolingTerm = absPooledActivations.sum()
        poolingCost = self.lambd * poolingTerm

        # Total cost
        cost = reconstructionCost + poolingCost

        # Gradient of reconstruction cost term
        RxT = dot(reconDiff, data.T)
        reconstructionCostGrad = 2 * dot(RxT + RxT.T, WW.T).T

        # Gradient of sparsity / pooling term
        SLOW_WAY = False
        if SLOW_WAY:
            poolingCostGrad = zeros(WW.shape)
            for ii in range(nDatapoints):
                for jj in range(self.HH.shape[0]):
                    poolingCostGrad += outer(1/absPooledActivations[jj, ii] * data[:,ii], (hidden[:,ii] * self.HH[jj,:])).T
            poolingCostGrad *= self.lambd
            print 'slow way'
            print poolingCostGrad[:4,:4]

        # fast way?
        Ha = dot(self.HH.T, 1/absPooledActivations)
        poolingCostGrad = self.lambd * dot(hidden * Ha, data.T)
        #print 'fast way'
        #print poolingCostGrad[:4,:4]

        WGrad = reconstructionCostGrad + poolingCostGrad

        # Log some statistics
        thislog = array([poolingCost, reconstructionCost, cost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        grad = l2RowScaledGrad(WWold, WW, WGrad)
        grad = grad.flatten()

        #pdb.set_trace()
        if self.float32:
            # convert back to keep fortran happy
            return cost, array(grad, dtype='float64')
        else:
            return cost, grad



if __name__ == '__main__':
    resman.start('tica_speed64', diary = True)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    
    random.seed(0)
    tica = TICA(imgShape = (16, 16),
                hiddenLayerShape = (20, 20),
                shrink = 0,
                lambd = .05,
                epsilon = 1e-5,
                float32 = False,
                saveDir = resman.rundir)
    tica.run(data, maxFun = 10)

    resman.stop()
