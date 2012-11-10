#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os
from numpy import *
from matplotlib import pyplot

from util.ResultsManager import resman
from util.dataLoaders import loadFromPklGz, saveToFile
from rica import RICA, l2RowScaled, l2RowScaledGrad

try:
    from numdifftools import Gradient
except ImportError:
    print 'Could not import numdifftools. Probably fine.'



def neighborMatrix(hiddenLayerShape, neighborhoodSize, shrink = 0, gaussian = False):
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

    if gaussian:
        sigmaSq = float(neighborhoodSize)**2
        halfIWidth = hiddenLayerShape[0]/2
        halfJWidth = hiddenLayerShape[1]/2
        for ii in range(pooledLayerShape[0]):
            for jj in range(pooledLayerShape[1]):
                for nnii in range(hiddenLayerShape[0]):
                    for nnjj in range(hiddenLayerShape[1]):
                        # get min dist (including wraparound)
                        iDiff = (((ii - (nnii-shrink))+halfIWidth) % hiddenLayerShape[0])-halfIWidth
                        jDiff = (((jj - (nnjj-shrink))+halfJWidth) % hiddenLayerShape[1])-halfJWidth
                        distSq = iDiff ** 2 + jDiff ** 2
                        weight = exp(-distSq / sigmaSq)
                        #print ii, jj, nnii, nnjj, '  ', iDiff, jDiff, distSq, '%.3f' % exp(-distSq / sigmaSq)
                        # cut off very low values
                        ret[ii, jj, nnii, nnjj] = weight if weight > .01 else 0
                        #ret[ii, jj, nnii, nnjj] = distSq   # Just for testing
    else:
        rangeNeighborII = range(-neighborhoodSize, neighborhoodSize + 1)
        rangeNeighborJJ = range(-neighborhoodSize, neighborhoodSize + 1)

        for ii in range(pooledLayerShape[0]):
            for jj in range(pooledLayerShape[1]):
                for nnii in rangeNeighborII:
                    for nnjj in rangeNeighborJJ:
                        ret[ii, jj, (ii+shrink+nnii) % hiddenLayerShape[0], (jj+shrink+nnjj) % hiddenLayerShape[1]] = 1

    ret = reshape(ret, (nPooled, nHidden))
    ret = (ret.T / sum(ret, 1)).T     # Normalize to total weight 1 per pooling unit
    return ret



def fullNeighborMatrix(hiddenLayerShape, neighborhoodSize):
    return neighborMatrix(hiddenLayerShape, neighborMatrix, shrink = 0)



class TICA(RICA):
    '''See RICA for constructor arguments.'''

    def __init__(self, imgShape, lambd = .005, hiddenLayerShape = (10,10), neighborhoodParams = ('gaussian', 1.0, 0),
                 epsilon = 1e-5, saveDir = '', float32 = False):
        ''''''
        self.hiddenLayerShape = hiddenLayerShape
        
        super(TICA, self).__init__(imgShape = imgShape,
                                   lambd = lambd,
                                   nFeatures = prod(self.hiddenLayerShape),
                                   epsilon = epsilon,
                                   float32 = float32,
                                   saveDir = saveDir)

        # Pooling neighborhood params
        if len(neighborhoodParams) != 3:
            raise Exception('Expected tuple of length 3 for neighborhoodParams')
        self.neighborhoodType, self.neighborhoodSize, self.shrink = neighborhoodParams
        self.neighborhoodType = self.neighborhoodType.lower()
        if self.neighborhoodType not in ('gaussian', 'flat'):
            raise Exception('Expected neighborhoodType to be gaussian or flat, but got "%s"' % repr(self.neighborhoodType))
        self.neighborhoodIsGaussian = (self.neighborhoodType == 'gaussian')

        self.nPooled = (self.hiddenLayerShape[0] - self.shrink*2) * (self.hiddenLayerShape[1] - self.shrink*2)
        self.HH = neighborMatrix(self.hiddenLayerShape, self.neighborhoodSize,
                                 shrink = self.shrink, gaussian = self.neighborhoodIsGaussian)

        if self.float32:
            self.HH = array(self.HH, dtype='float32')


    def cost(self, WW, data, plotEvery = None):
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

        numEvals = 0 if self.costLog is None else self.costLog.shape[0]
        if plotEvery and numEvals % plotEvery == 0:
            self.plotWW(WW, filePrefix = 'intermed_WW_%04d' % numEvals)

        # Forward Prop
        hidden = dot(WW, data)
        reconstruction = dot(WW.T, hidden)
        
        # Reconstruction cost
        reconDiff = reconstruction - data
        reconstructionCost = sum(reconDiff ** 2)

        # L2 Pooling / Sparsity cost
        absPooledActivations = sqrt(self.epsilon + dot(self.HH, hidden ** 2))
        poolingTerm = absPooledActivations.sum()
        poolingCost = self.lambd * poolingTerm

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

        # Total cost and gradient per training example
        poolingCost /= nDatapoints
        reconstructionCost /= nDatapoints
        totalCost = reconstructionCost + poolingCost
        reconstructionCostGrad /= nDatapoints
        poolingCostGrad /= nDatapoints
        WGrad = reconstructionCostGrad + poolingCostGrad

        # Log some statistics
        thislog = array([poolingCost, reconstructionCost, totalCost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        grad = l2RowScaledGrad(WWold, WW, WGrad)
        grad = grad.flatten()

        print 'f =', totalCost, '|grad| =', linalg.norm(grad)

        if self.float32:
            # convert back to keep fortran happy
            return totalCost, array(grad, dtype='float64')
        else:
            return totalCost, grad


    def getXYNumTiles(self):
        return self.hiddenLayerShape


    def plotCostLog(self):
        # plot sparsity/reconstruction costs over time
        costs = self.costLog
        self.costLog = None
        pyplot.plot(costs[:,0], 'b-', costs[:,1], 'r-')
        pyplot.hold(True)
        pyplot.plot(costs[:,2], '--', color = (.7,0,.7,1))
        pyplot.legend(('pooling/sparsity * %s' % repr(self.lambd), 'reconstruction', 'total'))
        pyplot.xlabel('iteration'); pyplot.ylabel('cost')
        if self.saveDir:
            pyplot.savefig(os.path.join(self.saveDir, 'cost.png'))
            pyplot.savefig(os.path.join(self.saveDir, 'cost.pdf'))



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    #data = data[:,:5000]  #HACK

    #hiddenISize = 20
    #hiddenJSize = 20
    #lambd = .05
    #neighborhoodSize = 1
    #print '\nChosen TICA parameters'

    hiddenISize = random.randint(4, 25+1)
    hiddenJSize = random.randint(10, 30+1)
    lambd = .05 * 2 ** random.randint(-4, 4+1)
    neighborhoodSize = random.uniform(.1,3)
    print '\nRandomly selected TICA parameters'

    for key in ['hiddenISize', 'hiddenJSize', 'lambd', 'neighborhoodSize']:
        print '  %20s: %s' % (key, locals()[key])
    
    random.seed(0)
    tica = TICA(imgShape = (16, 16),
                hiddenLayerShape = (hiddenISize, hiddenJSize),
                neighborhoodParams = ('gaussian', neighborhoodSize, 0),
                lambd = lambd,
                epsilon = 1e-5,
                float32 = False,
                saveDir = resman.rundir)
    tica.run(data, plotEvery = 5, maxFun = 300)

    resman.stop()
