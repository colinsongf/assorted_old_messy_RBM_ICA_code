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



def fullNeighborMatrix(hiddenLayerShape, neighborhoodSize):
    nHidden = prod(hiddenLayerShape)
    ret = zeros(hiddenLayerShape + hiddenLayerShape)  # 4D tensor

    rangeNeighborII = range(-neighborhoodSize, neighborhoodSize + 1)
    rangeNeighborJJ = range(-neighborhoodSize, neighborhoodSize + 1)

    for ii in range(hiddenLayerShape[0]):
        for jj in range(hiddenLayerShape[1]):
            for nnii in rangeNeighborII:
                for nnjj in rangeNeighborJJ:
                    ret[ii, jj, (ii+nnii) % hiddenLayerShape[0], (jj+nnjj) % hiddenLayerShape[1]] = 1

    return reshape(ret, (nHidden, nHidden))



def neighborMatrix(hiddenLayerShape, neighborhoodSize, shrink = 0):
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



class TICA(RICA):
    '''See RICA for constructor arguments.'''

    def __init__(self, imgShape, lambd = .005, hiddenLayerShape = (10,10), neighborhoodSize = 1, shrink = 0, epsilon = 1e-5, saveDir = ''):
        self.hiddenLayerShape = hiddenLayerShape
        
        super(TICA, self).__init__(imgShape = imgShape,
                                   lambd = lambd,
                                   nFeatures = prod(self.hiddenLayerShape),
                                   epsilon = epsilon,
                                   saveDir = saveDir)

        self.neighborhoodSize = neighborhoodSize
        self.shrink = shrink
        self.nPooled = (self.hiddenLayerShape[0] - self.shrink*2) * (self.hiddenLayerShape[1] - self.shrink*2)
        self.HH = neighborMatrix(self.hiddenLayerShape, self.neighborhoodSize, shrink = self.shrink)


    def cost(self, WW, data):
        '''Main method of TICA that differs from RICA.'''

        nInputDim = data.shape[0]
        nDatapoints = data.shape[1]
        if self.nInputDim != nInputDim:
            raise Exception('Expected shape %s = %d dimensional input, but got %d' % (repr(self.imgShape), self.nInputDim, nInputDim))

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

        # % Sparsity Cost
        # K = sqrt(params.epsilon + h.^2);
        # sparsity_cost = params.lambda * sum(sum(K));
        # K = 1./K;
        #KK = sqrt(self.epsilon + hidden ** 2)
        #sparsityCost = self.lambd * sum(KK)
        #KK = 1/KK

        #outDeriv = reconDiff

        # Total cost
        cost = reconstructionCost + poolingCost


        # HACK FOR TESTING!!!!!!!!!!!!!!
        #cost = reconstructionCost


        # Gradient of reconstruction cost term        
        RxT = dot(reconDiff, data.T)
        reconstructionCostGrad = 2 * dot(RxT + RxT.T, WW.T).T

        # Gradient of sparsity / pooling term
        SLOW_WAY = True
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
        poolingCostGrad = self.lambd * dot(data, (hidden * Ha).T).T
        #foo = self.lambd * ((1/absPooledActivations) * (hidden.T * self.HH))

        print 'fast way'
        print poolingCostGrad[:4,:4]




        oo = ones(nInputDim)
        gradMaybe = 2 * dot(dot(reconDiff.T, WW.T), (outer(oo, data) + outer(data, oo)))





        # Log some statistics
        thislog = array([poolingCost, reconstructionCost, cost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        return cost, gradMaybe    # HACK for now, don't return gradient
        return cost, None    # HACK for now, don't return gradient

    

        # % Backprop Output Layer
        # W2grad = outderv * h';
        W2Grad = dot(outDeriv, hidden.T)

        # % Backprop Hidden Layer
        # outderv = W * outderv;
        # outderv = outderv + params.lambda * (h .* K);
        outDeriv = dot(WW, outDeriv)
        outDeriv = outDeriv + self.lambd * (hidden * KK)

        # W1grad = outderv * x';
        # Wgrad = W1grad + W2grad';
        W1Grad = dot(outDeriv, data.T)
        WGrad = W1Grad + W2Grad.T

        # % unproject gradient for minFunc
        # grad = l2rowscaledg(Wold, W, Wgrad, 1);
        # grad = grad(:);
        grad = l2RowScaledGrad(WWold, WW, WGrad)
        grad = grad.flatten()

        print 'f =', cost, '|grad| =', linalg.norm(grad)

        return cost, grad

    def numericalCostGradient(self, WW, data):
        gradCost = dfun = Gradient(lambda w: self.cost(w, data))
        
        return gradCost(WW)



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    
    random.seed(0)
    tica = TICA(imgShape = (16, 16),
                hiddenLayerShape = (4, 5),
                lambd = .05,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    tica.run(data, maxFun = 300)

    resman.stop()
