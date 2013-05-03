#! /usr/bin/env python

import sys
import imp
import ipdb as pdb
import argparse
import types

#from utils import loadFromFile
#from squaresRbm import loadPickledData
from GitResultsManager import resman
#from util.plotting import plot3DShapeFromFlattened
from util.dataPrep import PCAWhiteningDataNormalizer  #, printDataStats



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
        return self._calculateSeesPatches(self, prevLayerSees, prevDistToNeighbor)

    def _calculateSeesPatches(self, prevLayerSees, prevDistToNeighbor):
        '''Default pass through version. Override in derived classes
        if desired. This method may assume both inputs are tuples.'''
        return prevLayerSees

    def calculateDistToNeighbor(self, prevDistToNeighbor):
        assert isinstance(prevDistToNeighbor, tuple), 'prevDistToNeighbor must be a tuple but it is %s' % repr(prevDistToNeighbor)
        return self._calculateDistToNeighbor(self, prevDistToNeighbor)

    def _calculateDistToNeighbor(self, prevDistToNeighbor):
        '''Default pass through version. Override in derived classes
        if desired. This method may assume prevDistToNeighbor is a
        tuple.'''
        return prevDistToNeighbor



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

        assert len(self.imageSize) == 2
        assert len(self.patchSize) == 2
        assert len(self.stride) == 2
        assert len(self.imageSize) == len(self.stride), 'imageSize and stride must be same length'

    def numPatches(self):
        '''How many patches fit within the data. Rounds down.'''
        return tuple([(ims-ps)/st+1 for ims,ps,st in zip(self.imageSize, self.patchSize, self.stride)])

    def getOutputSize(self):
        raise Exception('must implement in derived class')




class UpsonData3(DataLayer):

    def __init__(self, params):
        super(UpsonData3, self).__init__(params)

        self.colors = params['colors']

        assert self.colors in (1,3)

    def getOutputSize(self):
        if self.colors == 1:
            return self.patchSize
        else:
            return (self.patchSize[0], self.patchSize[1], 3)



######################
# Whitening
######################

class WhiteningLayer(NonDataLayer):

    trainable = True

    def __init__(self, params):
        super(WhiteningLayer, self).__init__(params)



class PCAWhiteningLayer(WhiteningLayer):

    def __init__(self, params):
        super(PCAWhiteningLayer, self).__init__(params)




######################
# Learning
######################

class TicaLayer(NonDataLayer):

    trainable = True

    def __init__(self, params):
        super(TicaLayer, self).__init__(params)

        self.hiddenSize = params['hiddenSize']
        self.neighborhood = params['neighborhood']
        self.lambd = params['lambd']
        self.epsilon = params['epsilon']

        assert isinstance(self.hiddenSize, tuple)
        assert len(self.neighborhood) == 4
        assert self.neighborhood[3] == 0   # shrink not supported yet (changes output size)

    def _calculateOutputSize(self, inputSize):
        return self.hiddenSize



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

class AvgPool(NonDataLayer):
    pass

class MaxPool(NonDataLayer):
    pass


class LcnLayer(NonDataLayer):

    def __init__(self, params):
        super(LcnLayer, self).__init__(params)



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