#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os
from numpy import *
from matplotlib import pyplot
from GitResultsManager import resman

from util.dataLoaders import loadFromPklGz, saveToFile
from util.cache import PersistentHasher
from rica import RICA, l2RowScaled, l2RowScaledGrad

try:
    from numdifftools import Gradient
except ImportError:
    print 'Could not import numdifftools. Probably fine.'



def neighborMatrix(hiddenLayerShape, neighborhoodSize, shrink = 0, gaussian = False, nPoolingIgnoredNeurons = 0):
    '''Generate the neighbor matrix H, a 4D tensor where for the original tensor:
    
        H_i,j,k,l = 1 if the pooled unit at i,j is connected to the hidden unit at k,l

    Note: the tensor is returned flattened to 2D.

    hiddenLayerShape: size of the hidden layer, like (20,30)
    neighborhoodSize: number of pixels to grow the neighborhood
        by. e.g. ns=0 -> no pooling, ns=1 -> 3x3 neighborhoods.
    shrink: shrink each edge of the pooling units by this much vs. the
        hidden layer size. Just added to illustrate that the number of
        pooling neurons need not equal the number of hidden neurons.
    nPoolingIgnoredNeurons: number of hidden neurons (starting at the
       top-left and continuing down the first column, then second
       column, and so on) to ignore when computing the pooled neuron
       responses. If nPoolingIgnoredNeurons > 0, we simply set that
       many corresponding entries in the neighborMatrix to 0.
    '''
    
    if shrink < 0:
        raise Exception('shrink parameter must be >= 0')
    if shrink*2 >= min(hiddenLayerShape):
        raise Exception('shrink parameter must be < min(hiddenLayerShape)/2')
        
    pooledLayerShape = (hiddenLayerShape[0]-2*shrink, hiddenLayerShape[1]-2*shrink)
    nHidden = prod(hiddenLayerShape)
    nPooled = prod(pooledLayerShape)

    if nPoolingIgnoredNeurons < 0 or nPoolingIgnoredNeurons > nHidden:
        raise Exception('Expected nPoolingIgnoredNeurons in [0, %d] but got %d' % (nHidden, nPoolingIgnoredNeurons))

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
    ret[:,0:nPoolingIgnoredNeurons] = 0     # Ignore some number of hidden neurons
    ret = (ret.T / sum(ret, 1)).T     # Normalize to total weight 1 per pooling unit
    return ret



def fullNeighborMatrix(hiddenLayerShape, neighborhoodSize):
    return neighborMatrix(hiddenLayerShape, neighborMatrix, shrink = 0)



class TICA(RICA):
    '''See RICA for constructor arguments.'''

    def __init__(self, nInputs, lambd = .005, hiddenLayerShape = (10,10), neighborhoodParams = ('gaussian', 1.0, 0, 0),
                 epsilon = 1e-5, saveDir = '', float32 = False, initWW = True):
        ''''''
        self.hiddenLayerShape = hiddenLayerShape
        self.nHidden = prod(self.hiddenLayerShape)
        
        super(TICA, self).__init__(nInputs = nInputs,
                                   nOutputs = self.nHidden,
                                   lambd = lambd,
                                   epsilon = epsilon,
                                   float32 = float32,
                                   saveDir = saveDir,
                                   initWW = initWW)

        # Pooling neighborhood params
        # ('type', size, shrink, ignore)
        if len(neighborhoodParams) != 4:
            raise Exception('Expected tuple of length 4 for neighborhoodParams')
        self.neighborhoodType, self.neighborhoodSize, self.shrink, self.nPoolingIgnoredNeurons = neighborhoodParams
        self.neighborhoodType = self.neighborhoodType.lower()
        if self.neighborhoodType not in ('gaussian', 'flat'):
            raise Exception('Expected neighborhoodType to be gaussian or flat, but got "%s"' % repr(self.neighborhoodType))
        self.neighborhoodIsGaussian = (self.neighborhoodType == 'gaussian')

        self.nPooled = (self.hiddenLayerShape[0] - self.shrink*2) * (self.hiddenLayerShape[1] - self.shrink*2)
        self.HH = neighborMatrix(self.hiddenLayerShape, self.neighborhoodSize,
                                 shrink = self.shrink, gaussian = self.neighborhoodIsGaussian,
                                 nPoolingIgnoredNeurons = self.nPoolingIgnoredNeurons)

        if self.float32:
            self.HH = array(self.HH, dtype='float32')


    def costAndLog(self, WW, data, plotEvery = None):
        totalCost, poolingCost, reconstructionCost, grad = self.cost(WW, data, plotEvery = plotEvery)

        # Log some statistics
        thislog = array([poolingCost, reconstructionCost, totalCost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        print 'f =', totalCost, '|grad| =', linalg.norm(grad)

        return totalCost, grad

    def cost(self, WW, data, plotEvery = None, returnFull = False):
        '''Main method of TICA that differs from RICA.

        if returnFull:
            returns totalCost, poolingCost, reconstructionCost, grad, hidden, reconDiff
        else:
            returns totalCost, poolingCost, reconstructionCost, grad
        '''

        #pdb.set_trace()

        nInputs = data.shape[0]
        nDatapoints = data.shape[1]
        if self.nInputs != nInputs:
            raise Exception('Expected %d dimensional input, but got %d' % (self.nInputs, nInputs))

        if self.float32:
            WW = array(WW, dtype='float32')
        
        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nHidden, nInputs)
        WWold = WW
        WW = l2RowScaled(WW)

        numEvals = 0 if self.costLog is None else self.costLog.shape[0]
        if plotEvery and numEvals % plotEvery == 0:
            self.plotWW(WW, filePrefix = 'intermed_WW_%04d' % numEvals)

        #from time import time
        #pdb.set_trace()

        # Forward Prop
        hidden = dot(WW, data)                   # 4.0s, aligned (C * C)
        reconstruction = dot(WW.T, hidden)       # 6.3s, misaligned: (F * C)
        
        # Reconstruction cost
        reconDiff = reconstruction - data
        reconstructionCost = sum(reconDiff ** 2)

        # L2 Pooling / Sparsity cost
        absPooledActivations = sqrt(self.epsilon + dot(self.HH, hidden ** 2))     # 2.9s, aligned (C * C)
        poolingTerm = absPooledActivations.sum()
        poolingCost = self.lambd * poolingTerm

        # Gradient of reconstruction cost term
        RxT = dot(reconDiff, data.T)                                              # 1.7s, misaligned (C * F)
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

        # fast way
        Ha = dot(self.HH.T, 1/absPooledActivations)                              # 3.1s, misaligned (F * C) -> -0.3 with self.HHT
        poolingCostGrad = self.lambd * dot(hidden * Ha, data.T)                  # 1.5s, misaligned (C * F) -> -0.3 with dataT
        #print 'fast way'
        #print poolingCostGrad[:4,:4]

        # Total cost and gradient per training example
        poolingCost /= nDatapoints
        reconstructionCost /= nDatapoints
        totalCost = reconstructionCost + poolingCost
        reconstructionCostGrad /= nDatapoints
        poolingCostGrad /= nDatapoints
        WGrad = reconstructionCostGrad + poolingCostGrad

        grad = l2RowScaledGrad(WWold, WW, WGrad)
        grad = grad.flatten()

        if self.float32:
            # convert back to keep fortran happy
            grad = array(grad, dtype='float64')

        if returnFull:
            return totalCost, poolingCost, reconstructionCost, grad, hidden, reconDiff
        else:
            return totalCost, poolingCost, reconstructionCost, grad


    def getRepresentation(self, data):
        '''Assumes data is one example per column. Returns '''
        #pdb.set_trace()

        nInputs = data.shape[0]
        if self.nInputs != nInputs:
            raise Exception('Expected %d dimensional input, but got %d' % (self.nInputs, nInputs))
        if self.float32:
            WW = array(self.WW, dtype='float32')
        else:
            WW = self.WW
        
        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nHidden, nInputs)
        WW = l2RowScaled(WW)

        # Forward Prop
        hidden = dot(WW, data)
        absPooledActivations = sqrt(self.epsilon + dot(self.HH, hidden ** 2))

        return hidden, absPooledActivations


    def getXYNumTiles(self):
        return self.hiddenLayerShape


    def plotCostLog(self, saveDir = None, prefix = None):
        # plot sparsity/reconstruction costs over time
        costs = self.costLog
        #self.costLog = None     # disabled
        pyplot.figure()
        pyplot.plot(costs[:,0], 'b-', costs[:,1], 'r-')
        pyplot.hold(True)
        pyplot.plot(costs[:,2], '--', color = (.7,0,.7,1))
        pyplot.legend(('pooling/sparsity * %s' % repr(self.lambd), 'reconstruction', 'total'))
        pyplot.xlabel('iteration'); pyplot.ylabel('cost')
        if saveDir is None:
            saveDir = self.saveDir
        if saveDir:
            pyplot.savefig(os.path.join(saveDir, (prefix if prefix else '') + 'cost.png'))
            pyplot.savefig(os.path.join(saveDir, (prefix if prefix else '') + 'cost.pdf'))
        pyplot.close()


    def getReconPlotString(self, costEtc):
        totalCost, poolingCost, reconstructionCost, grad = costEtc
        return 'R: %g P*%g: %g T: %g' % (reconstructionCost, self.lambd, poolingCost, totalCost)


    def __hash__(self):
        hasher = PersistentHasher()
        hasher.update('TICA')
        hasher.update(self.nInputs)
        hasher.update(self.nOutputs)
        hasher.update(self.lambd)
        hasher.update(self.epsilon)
        hasher.update(self.float32)
        hasher.update(self.WWshape)
        hasher.update(self.WW)
        hasher.update(self.hiddenLayerShape)
        hasher.update(self.nHidden)
        hasher.update(self.neighborhoodType)
        hasher.update(self.neighborhoodSize)
        hasher.update(self.shrink)
        hasher.update(self.nPoolingIgnoredNeurons)
        hasher.update(self.neighborhoodIsGaussian)
        hasher.update(self.nPooled)
        hasher.update(self.HH)
        return int(hasher.hexdigest(), 16)
        #return int(hasher.hexdigest()[:7], 16)  # only 7 hex digits fit into an int

    def __cmp__(self, other):
        return self.__hash__() - other.__hash__()



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
    lambd = .1 * 2 ** random.randint(-5, 5+1)
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
    tica.run(data, plotEvery = 2, maxFun = 300)

    resman.stop()
