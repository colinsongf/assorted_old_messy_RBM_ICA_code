#! /usr/bin/env python

import sys
import imp
import ipdb as pdb
import argparse
import types
import time
from numpy import *

from util.cache import cached, PersistentHasher
from GitResultsManager import resman
from util.dataPrep import PCAWhiteningDataNormalizer  #, printDataStats
from makeData import makeUpsonRovio3
from tica import TICA, neighborMatrix



class DataArrangement(object):
    '''Represents a particular arragement of data. Example: 1000 layers of 2 x
    3 patches each, with each patch of unrolled length N, could be
    represented as an N x 6000 matrix. In this case, we would use

    DataArrangement((2,3), 1000)
    '''
    
    def __init__(self, layerShape, nLayers):
        self.layerShape = layerShape      # number of patches along i, j, ... e.g.: (2,3)
        self.nLayers    = nLayers         # number of layers, e.g.: 1000

    def __repr__(self):
        return 'DataArrangement(layerShape = %s, nLayers = %s)' % (repr(self.layerShape), repr(self.nLayers))

    def totalPatches(self):
        return prod(self.layerShape) * self.nLayers



######################
# Base classes
######################

class Layer(object):

    trainable = False      # default
    
    def __init__(self, params):
        self.name = params['name']
        self.layerType = params['type']

        # Calculated when layer is created
        self.inputSize = None     # size of input
        self.outputSize = None    # size of output



class NonDataLayer(Layer):

    isDataLayer = False

    def __init__(self, params):
        super(NonDataLayer, self).__init__(params)

        # Here we keep track of two important quantities:
        # 1. how many patches each layer can see (needed when
        #    determining how large of data samples are needed for
        #    training this layer), and
        # 2. the distance, in number of patchs, to the nearest
        #    neighbor in both i and j. This is needed for calculating
        #    (1) on higher layers.
        self.seesPatches = None      # tuple, number of patches in the data layer this layer is exposed to
        self.distToNeighbor = None   # tuple, number of patches that the nearest neighbor is in i and j

    def calculateOutputSize(self, inputSize):
        assert isinstance(inputSize, tuple), 'inputSize must be a tuple but it is %s' % repr(inputSize)
        return self._calculateOutputSize(inputSize)

    def _calculateOutputSize(self, inputSize):
        '''Default pass through version. Override in derived classes
        if desired. This method may assume inputSize is a tuple.'''
        return inputSize

    def calculateSeesPatches(self, prevLayerSees, prevDistToNeighbor):
        assert isinstance(prevLayerSees, tuple), 'prevLayerSees must be a tuple but it is %s' % repr(prevLayerSees)
        assert isinstance(prevDistToNeighbor, tuple), 'prevDistToNeighbor must be a tuple but it is %s' % repr(prevDistToNeighbor)
        assert len(prevLayerSees) == len(prevDistToNeighbor)
        return self._calculateSeesPatches(prevLayerSees, prevDistToNeighbor)

    def _calculateSeesPatches(self, prevLayerSees, prevDistToNeighbor):
        '''Default pass through version. Override in derived classes
        if desired. This method may assume both inputs are tuples.'''
        return prevLayerSees

    def calculateDistToNeighbor(self, prevDistToNeighbor):
        assert isinstance(prevDistToNeighbor, tuple), 'prevDistToNeighbor must be a tuple but it is %s' % repr(prevDistToNeighbor)
        return self._calculateDistToNeighbor(prevDistToNeighbor)

    def _calculateDistToNeighbor(self, prevDistToNeighbor):
        '''Default pass through version. Override in derived classes
        if desired. This method may assume prevDistToNeighbor is a
        tuple.'''
        return prevDistToNeighbor

    def forwardProp(self, data, dataArrangement):
        '''
        Input:
        data - one example per column

        dataArrangement - specifies how the data should be interpreted.
        Many classes will not care about the arrangement
        of the data, in which case they can ignore dataArrangement

        Returns:
        representation, newDataArrangement
        '''
        
        if self.trainable and not self.isInitialized:
            raise Exception('Must initialize %s layer first' % self.name)
        if self.trainable and not self.isTrained:
            print 'WARNING: forwardProp through untrained layer, might not be desired'
        inDimension, numExamples = data.shape
        if inDimension != prod(self.inputSize):
            raise Exception('Layer %s expects examples of shape %s = %s rows but got %s data matrix'
                            % (self.name, self.inputSize, prod(self.inputSize), data.shape))
        self._checkDataArrangement(data, dataArrangement)

        representation, newDataArrangement = self._forwardProp(data, dataArrangement)

        self._checkDataArrangement(representation, newDataArrangement)
        if len(dataArrangement.layerShape) != len(newDataArrangement.layerShape):
            raise Exception('Conversion from %s to %s is invalid'
                            % (dataArrangement, newDataArrangement))
        if representation.shape[0] != prod(self.outputSize):
            raise Exception('Layer %s was supposed to output examples of shape %s = %s rows but got %s output matrix'
                            % (self.name, self.outputSize, prod(self.outputSize), representation.shape))
        return representation, newDataArrangement

    def _forwardProp(self, data, dataArrangement):
        '''Default pass through version. Override in derived classes
        if desired.'''
        return data, dataArrangement

    def _checkDataArrangement(self, data, dataArrangement):
        if dataArrangement.totalPatches() != data.shape[1]:
            raise Exception('dataArrangement mismatched with data (data.shape is %s, dataArrangement is %s, %d != %d'
                            % (data.shape, dataArrangement, data.shape[1], dataArrangement.totalPatches()))



class TrainableLayer(NonDataLayer):

    trainable = True

    def __init__(self, params):
        super(TrainableLayer, self).__init__(params)
        self.isInitialized = False
        self.isTrained = False

    def initialize(self):
        if self.isInitialized:
            raise Exception('Layer was already initialized')
        self._initialize()
        self.isInitialized = True

    def _initialize(self):
        '''Default no-op version. Override in derived class.'''
        pass

    def train(self, data, dataArrangement, trainParams = None):
        if self.isTrained:
            raise Exception('Layer was already trained')
        self._checkDataArrangement(data, dataArrangement)
        self._train(data, dataArrangement, trainParams)
        self.isTrained = True

    def _train(self, data, dataArrangement, trainParams = None):
        '''Default no-op version. Override in derived class.'''
        pass
        



######################
# Data
######################

class DataLayer(Layer):

    isDataLayer = True

    def __init__(self, params):
        super(DataLayer, self).__init__(params)

        self.imageSize = params['imageSize']
        self.patchSize = params['patchSize']
        self.stride = params['stride']

        # Only 2D data types for now (color is fine though)
        assert len(self.imageSize) == 2
        assert len(self.patchSize) == 2
        assert len(self.stride) == 2
        assert len(self.imageSize) == len(self.stride), 'imageSize and stride must be same length'

    def numPatches(self):
        '''How many patches fit within the data. Rounds down.'''
        return tuple([(ims-ps)/st+1 for ims,ps,st in zip(self.imageSize, self.patchSize, self.stride)])

    def getOutputSize(self):
        raise Exception('must implement in derived class')

    def getData(self):
        raise Exception('must implement in derived class')



class UpsonData3(DataLayer):

    def __init__(self, params):
        super(UpsonData3, self).__init__(params)

        self.colors = params['colors']
        #self.rng = random.RandomState()   # takes seed from timer initially

        assert self.colors in (1,3)

    def getOutputSize(self):
        if self.colors == 1:
            return self.patchSize
        else:
            return (self.patchSize[0], self.patchSize[1], 3)

    def getData(self, patchSize, number, seed = None):
        samples, labelMatrix, labelStrings = cached(makeUpsonRovio3.randomSampleMatrixWithLabels,
                                                    makeUpsonRovio3.trainFilter,
                                                    color = (self.colors == 3),
                                                    Nw = patchSize, Nsamples = number, seed = seed,
                                                    imgDirectory = '../data/upson_rovio_3/imgfiles')

        return samples.T    # one example per column



######################
# Whitening
######################

class WhiteningLayer(TrainableLayer):

    def __init__(self, params):
        super(WhiteningLayer, self).__init__(params)



class PCAWhiteningLayer(WhiteningLayer):

    def __init__(self, params):
        super(PCAWhiteningLayer, self).__init__(params)
        self.pcaWhiteningDataNormalizer = None

    def _train(self, data, dataArrangement, trainParams = None):
        self.pcaWhiteningDataNormalizer = PCAWhiteningDataNormalizer(data)

    def _forwardProp(self, data, dataArrangement):
        dataWhite, junk = self.pcaWhiteningDataNormalizer.raw2normalized(data, unitNorm = True)
        return dataWhite, dataArrangement



######################
# Learning
######################

class TicaLayer(TrainableLayer):

    def __init__(self, params):
        super(TicaLayer, self).__init__(params)

        self.hiddenSize = params['hiddenSize']
        self.neighborhood = params['neighborhood']
        self.lambd = params['lambd']
        self.epsilon = params['epsilon']
        self.tica = None

        assert isinstance(self.hiddenSize, tuple)
        assert len(self.neighborhood) == 4
        assert self.neighborhood[3] == 0   # shrink not supported yet (changes output size)

    def _calculateOutputSize(self, inputSize):
        return self.hiddenSize

    def _train(self, data, dataArrangement, trainParams):
        logDir = trainParams.get('logDir', None)
        # Learn model
        tica = TICA(nInputs            = prod(self.inputSize),
                    hiddenLayerShape   = self.hiddenSize,
                    neighborhoodParams = self.neighborhood,
                    lambd              = self.lambd,
                    epsilon            = self.epsilon)

        beginTotalCost, beginPoolingCost, beginReconstructionCost, grad = tica.cost(tica.WW, data)

        tic = time.time()
        tica.learn(data, maxFun = trainParams['maxFuncCalls'])
        execTime = time.time() - tic
        if logDir:
            saveToFile(os.path.join(logDir, 'tica.pkl.gz'), tica)    # save learned model

        endTotalCost, endPoolingCost, endReconstructionCost, grad = tica.cost(tica.WW, data)

        print 'beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime ='
        print [beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime]

        # Plot some results
        #plotImageRicaWW(tica.WW, imgShape, saveDir, tileShape = hiddenLayerShape, prefix = pc('WW_iterFinal'))
        if logDir:
            tica.plotResults(logDir, tica, data, self.numInputs, self.hiddenSize)

        self.tica = tica


    def _forwardProp(self, data, dataArrangement):
        hidden, absPooledActivations = self.tica.getRepresentation(data)
        return absPooledActivations, dataArrangement



######################
# Downsampling, LCN, Concatenation
######################

class DownsampleLayer(NonDataLayer):

    def __init__(self, params):
        super(DownsampleLayer, self).__init__(params)

        self.factor = params['factor']

        assert len(self.factor) in (1,2)

    def _calculateOutputSize(self, inputSize):
        '''rounds down'''
        assert len(inputSize) <= 3
        if len(inputSize) == 3:
            # color images
            assert len(self.factor) == 2, 'for color images, self.factor must be length 2 (i and j factors)'
            # don't downsample third dimensions (number of colors, probably 3)
            return (inputSize[0]/self.factor[0], inputSize[1]/self.factor[1], inputSize[2])
        else:
            # linear data or single-channel images
            assert len(self.factor) == len(inputSize), 'for inputSize length 1 or 2, self.factor must match length of inputSize'
            if len(inputSize) == 1:
                return (inputSize[0]/self.factor[0],)
            else: # length 2
                return (inputSize[0]/self.factor[0], inputSize[1]/self.factor[1])

    def _forwardProp(self, data, dataArrangement):
        dimension, numExamples = data.shape

        patches = reshape(data, self.inputSize + (numExamples,))
        # len(self.factor) is either 1 or 2
        if len(self.factor) == 1:
            downsampled = patches[::self.factor[0],:]
        else:
            downsampled = patches[::self.factor[0],::self.factor[1],:]
            
        output = reshape(downsampled, (prod(self.outputSize), numExamples))
        return output, dataArrangement



class AvgPool(NonDataLayer):
    pass

class MaxPool(NonDataLayer):
    pass


class LcnLayer(NonDataLayer):

    def __init__(self, params):
        super(LcnLayer, self).__init__(params)
        self.gaussWidth = params['gaussWidth']

    def _forwardProp(self, data, dataArrangement):
        dimension, numExamples = data.shape

        gaussNeighbors = neighborMatrix(self.inputSize, self.gaussWidth, gaussian=True)

        # 2. LCN
        vv = data - dot(gaussNeighbors, data)
        sig = sqrt(dot(gaussNeighbors, vv**2))
        cc = .01     # ss = sorted(sig.flatten()); ss[len(ss)/10] = 0.026 in one test. So .01 seems about right.
        yy = vv / maximum(cc, sig)

        return yy, dataArrangement



class ConcatenationLayer(NonDataLayer):

    def __init__(self, params):
        super(ConcatenationLayer, self).__init__(params)

        self.concat = params['concat']
        self.stride = params['stride']

        assert isinstance(self.concat, tuple)
        assert isinstance(self.stride, tuple)
        assert len(self.concat) == len(self.stride)
        for ii, st in enumerate(self.stride):
            assert st >= 1 and st <= self.concat[ii], 'each element of stride must be at least one and at most %s' % self.concat

    def _calculateOutputSize(self, inputSize):
        if len(inputSize) == 3:
            # color images
            assert len(self.concat) == 2, 'for color images, self.concat must be length 2 (i and j concat factors)'
            # don't touch third dimensions (number of colors, probably 3)
            return (inputSize[0]*self.concat[0], inputSize[1]*self.concat[1], inputSize[2])
        else:
            # linear data or single-channel images
            assert len(self.concat) == len(inputSize), 'for inputSize length 1 or 2, self.concat must match length of inputSize'
            if len(inputSize) == 1:
                return (inputSize[0]*self.concat[0],)
            else: # length 2
                return (inputSize[0]*self.concat[0], inputSize[1]*self.concat[1])

    def _calculateSeesPatches(self, prevLayerSees, prevDistToNeighbor):
        assert len(prevLayerSees) == len(self.concat)
        if len(self.concat) == 1:
            return (prevLayerSees[0] + (self.concat[0]-1) * prevDistToNeighbor[0],)
        elif len(self.concat) == 2:
            return (prevLayerSees[0] + (self.concat[0]-1) * prevDistToNeighbor[0],
                    prevLayerSees[1] + (self.concat[1]-1) * prevDistToNeighbor[1])
        else:
            raise Exception('logic error')

    def _calculateDistToNeighbor(self, prevDistToNeighbor):
        assert len(prevDistToNeighbor) == len(self.concat)
        if len(self.concat) == 1:
            return (prevDistToNeighbor[0] * self.stride[0],)
        elif len(self.concat) == 2:
            return (prevDistToNeighbor[0] * self.stride[0],
                    prevDistToNeighbor[1] * self.stride[1])
        else:
            raise Exception('logic error')

    def _forwardProp(self, data, dataArrangement):
        '''This is where the concatenation actually takes place.'''

        newLayerShape = tuple([1+(layerShape - concat) / stride for layerShape,concat,stride
                               in zip(dataArrangement.layerShape, self.concat, self.stride)])
        remainders    = tuple([(layerShape - concat) % stride for layerShape,concat,stride
                               in zip(dataArrangement.layerShape, self.concat, self.stride)])
        tooSmall = sum([nls < 1 for nls in newLayerShape])  # must be at least 1 in every dimension
        if tooSmall or any(remainders):
            raise Exception('%s cannot be evenly concatenated by concat %s and stride %s'
                            % (dataArrangement, self.concat, self.stride))

        reshapedData = reshape(data, (data.shape[0], dataArrangement.nLayers,) + dataArrangement.layerShape)

        newDataArrangement = DataArrangement(newLayerShape, dataArrangement.nLayers)

        # BEGIN: 2D data assumption
        # Note: this only works for 2D data! Add other cases if needed.

        assert len(self.concat) == 2, 'Only works for 2D data for now!'
        assert len(dataArrangement.layerShape) == 2, 'Only works for 2D data for now!'

        concatenatedData = zeros((prod(self.outputSize),
                                  prod(newLayerShape) * dataArrangement.nLayers))
        
        oldPatchLength = prod(self.inputSize)
        newPatchCounter = 0
        # Note: this code assumes the default numpy flattening order:
        # C-order, or row-major order.
        for layerIdx in xrange(dataArrangement.nLayers):
            for newPatchII in xrange(newLayerShape[0]):
                for newPatchJJ in xrange(newLayerShape[1]):
                    oldPatchStartII = newPatchII * self.stride[0]
                    oldPatchStartJJ = newPatchJJ * self.stride[1]
                    curLoc = 0
                    for oldPatchII in xrange(oldPatchStartII, oldPatchStartII + self.concat[0]):
                        for oldPatchJJ in xrange(oldPatchStartJJ, oldPatchStartJJ + self.concat[1]):
                            concatenatedData[curLoc:(curLoc+oldPatchLength),newPatchCounter] = \
                              reshapedData[:,layerIdx,oldPatchII,oldPatchJJ]
                            curLoc += oldPatchLength
                    newPatchCounter += 1
        # sanity check
        assert newPatchCounter == concatenatedData.shape[1]
        assert curLoc == prod(self.outputSize)
        # END: 2D data assumption

        return concatenatedData, newDataArrangement
