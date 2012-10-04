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



def neighborMatrix(hiddenLayerShape, neighborhoodSize):
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



class TICA(RICA):
    '''See RICA for constructor arguments.'''

    def __init__(self, imgShape, lambd = .005, hiddenLayerShape = (10,10), neighborhoodSize = 1, epsilon = 1e-5, saveDir = ''):
        self.hiddenLayerShape = hiddenLayerShape
        
        super(TICA, self).__init__(imgShape = imgShape,
                                   lambd = lambd,
                                   nFeatures = prod(self.hiddenLayerShape),
                                   epsilon = epsilon,
                                   saveDir = saveDir)

        self.neighborhoodSize = neighborhoodSize
        self.HH = neighborMatrix(self.hiddenLayerShape, self.neighborhoodSize)
        pdb.set_trace()


    def cost(self, WW, data):
        '''Only method of TICA that is different.'''

        nInputDim = data.shape[0]
        if self.nInputDim != nInputDim:
            raise Exception('Expected shape %s = %d dimensional input, but got %d' % (repr(self.imgShape), self.nInputDim, nInputDim))

        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nFeatures, nInputDim)
        WWold = WW
        WW = l2RowScaled(WW)

        pdb.set_trace()
        print 'TICA HERE'

        # % Forward Prop
        # h = W*x;
        # r = W'*h;
        hidden = dot(WW, data)
        reconstruction = dot(WW.T, hidden)
        
        # % Sparsity Cost
        # K = sqrt(params.epsilon + h.^2);
        # sparsity_cost = params.lambda * sum(sum(K));
        # K = 1./K;
        #KK = sqrt(self.epsilon + hidden ** 2)
        #sparsityCost = self.lambd * sum(KK)
        #KK = 1/KK

        # % Reconstruction Loss and Back Prop
        # diff = (r - x);
        # reconstruction_cost = 0.5 * sum(sum(diff.^2));
        # outderv = diff;
        reconDiff = reconstruction - data
        reconstructionCost = .5 * sum(reconDiff ** 2)
        outDeriv = reconDiff

        # % compute the cost comprised of: 1) sparsity and 2) reconstruction
        # cost = sparsity_cost + reconstruction_cost;
        #print '   sp', sparsityCost, 'rc', reconstructionCost
        cost = sparsityCost + reconstructionCost

        thislog = array([sparsityCost, reconstructionCost, cost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

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
