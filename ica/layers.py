#! /usr/bin/env python

import sys
import imp
#import pdb
import ipdb as pdb
import argparse
import types
import time
import os
from IPython import embed
from matplotlib import pyplot
import matplotlib.cm as cm
from numpy import *
from scipy.optimize import minimize

from util.cache import cached, PersistentHasher, persistentHash
from util.misc import invSigmoid01
from GitResultsManager import resman, fmtSeconds
from util.dataPrep import PCAWhiteningDataNormalizer
from util.dataLoaders import loadNYU2Data, loadCS294Images
from makeData import makeUpsonRovio3
from tica import TICA, neighborMatrix
from cost import autoencoderCost, autoencoderForwardprop, autoencoderBackprop
from visualize import plotImageData



MAX_CACHE_SIZE_MB = 500

class DataArrangement(object):
    '''Represents a particular arragement of data. Example: 1000 slices of 2 x
    3 patches each, with each patch of unrolled length N, could be
    represented as an N x 6000 matrix. In this case, we would use

    DataArrangement((2,3), 1000)
    '''
    
    def __init__(self, sliceShape, nSlices):
        self.sliceShape = sliceShape      # number of patches along i, j, ... e.g.: (2,3)
        self.nSlices    = nSlices         # number of layers, e.g.: 1000

    def __repr__(self):
        return 'DataArrangement(sliceShape = %s, nSlices = %s)' % (repr(self.sliceShape), repr(self.nSlices))

    def totalPatches(self):
        return prod(self.sliceShape) * self.nSlices



######################
# Base Layer classes
######################

class Layer(object):

    trainable = False      # default
    nSublayers = 1         # default
    
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

    def forwardProp(self, data, dataArrangement, sublayer = None, quiet = False, outputBpHint = False):
        '''
        Input:
        data - one example per column (if single dim: promoted to two)

        dataArrangement - specifies how the data should be interpreted.
        Many classes will not care about the arrangement
        of the data, in which case they can ignore dataArrangement

        Returns:
        representation, newDataArrangement
        '''
        
        if self.trainable and not self.isInitialized:
            raise Exception('Must initialize %s layer first' % self.name)
        if (self.trainable and not self.isTrained) and not quiet:
            print 'WARNING: forwardProp through untrained layer, might not be desired'
        if sublayer is None: sublayer = self.nSublayers-1  # prop through all sublayers by default
        if sublayer not in range(self.nSublayers):
            raise Exception('sublayer must be None or in %s, but it is %s' % (repr(range(self.nSublayers)), repr(sublayer)))
        if len(data.shape) == 1:
            data = reshape(data, data.shape + (1,))    # promote to 2-dim at least
        inDimension, numExamples = data.shape
        if inDimension != prod(self.inputSize):
            raise Exception('Layer %s expects examples of shape %s = %s rows but got %s data matrix'
                            % (self.name, self.inputSize, prod(self.inputSize), data.shape))
        self._checkDataArrangement(data, dataArrangement)

        output = self._forwardProp(data, dataArrangement, sublayer, outputBpHint = outputBpHint)

        if outputBpHint:
            representation, newDataArrangement, bpHint = output
        else:
            representation, newDataArrangement = output
            
        self._checkDataArrangement(representation, newDataArrangement)
        if len(dataArrangement.sliceShape) != len(newDataArrangement.sliceShape):
            raise Exception('Conversion from %s to %s is invalid'
                            % (dataArrangement, newDataArrangement))
        if representation.shape[0] != prod(self.outputSize):
            raise Exception('Layer %s was supposed to output examples of shape %s = %s rows but got %s output matrix'
                            % (self.name, self.outputSize, prod(self.outputSize), representation.shape))
        return output

    def _forwardProp(self, data, dataArrangement, sublayer, outputBpHint):
        '''Default pass through version. Override in derived classes
        if desired.'''
        return data, dataArrangement

    def backProp(self, dqda, data, dataArrangement, sublayer = None, bpHint = None, verifyBpHint = False, quiet = False):
        '''
        Input:

        dqda - matrix of column vectors (single dim is promoted) of
               del q / del a_i, where q is some quantity and a_i
               is the ith output of this layer.

        data, dataArrangement - as in forward prop

        bpHint - the hint output by the layer on the forward pass,
                 if any. This data is specific to the layer and
                 should be considered to be an opaque token from the
                 outside (for NN: this could be the pre-sigmoid activations)

        verifyBpHint - if True, forwardProp the given data and check
                 that it produces the given hint

        Returns:
        dqdx - column vector (or matrix of column vectors) of
               del q / del x_i, where x_i is the ith input to
               this layer.
        '''
        
        if self.trainable and not self.isInitialized:
            raise Exception('Must initialize %s layer first' % self.name)
        if (self.trainable and not self.isTrained) and not quiet:
            print 'WARNING: backProp through untrained layer, might not be desired'
        if sublayer is None: sublayer = 0  # backprop through all sublayers by default
        if sublayer not in range(self.nSublayers):
            raise Exception('sublayer must be None or in %s, but it is %s' % (repr(range(self.nSublayers)), repr(sublayer)))
        # Additional check for now, because backprop to non-zero sublayer is not yet supported:
        if sublayer != 0:
            raise Exception('For now, backprop sublayer must be 0.')
        if len(data.shape) == 1:
            data = reshape(data, data.shape + (1,))    # promote to 2-dim at least
        inDimension, numExamples = data.shape
        if inDimension != prod(self.inputSize):
            raise Exception('Layer %s expects examples of shape %s = %s rows but got %s data matrix'
                            % (self.name, self.inputSize, prod(self.inputSize), data.shape))
        if len(dqda.shape) == 1:
            dqda = reshape(dqda, dqda.shape + (1,))    # promote to 2-dim at least
        outDimension, numExamples = dqda.shape
        if outDimension != prod(self.outputSize):
            raise Exception('Layer %s expects output examples of shape %s = %s rows but got %s dqda matrix'
                            % (self.name, self.outputSize, prod(self.outputSize), dqda.shape))
        self._checkDataArrangement(data, dataArrangement)

        # TODO: Mabe add something in about output activation shape?
        #self._checkDataArrangement(data, dataArrangement)

        if verifyBpHint:
            output = self.forwardProp(data, dataArrangement, outputBpHint = True, quiet = quiet)   # sublayer not supported yet
            representation, newDataArrangement, computedBpHint = output
            if persistentHash(bpHint) == persistentHash(computedBpHint):
                print 'layer:backProp: bpHint verified.'
            else:
                raise Exception('Given bpHint does not match computed bpHint.')

        dqdx, newDataArrangement = self._backProp(dqda, data, dataArrangement, bpHint)
        
        # TODO: probably add this back in somehow
        #self._checkDataArrangement(representation, newDataArrangement)
        if len(dataArrangement.sliceShape) != len(newDataArrangement.sliceShape):
            raise Exception('Conversion from %s to %s is invalid (TODO: check this. unsure about backprop)'
                            % (dataArrangement, newDataArrangement))
        if dqdx.shape[0] != prod(self.inputSize):
            raise Exception('Layer %s was supposed to outputdqdxs of shape %s = %s rows but got %s dqdx matrix'
                            % (self.name, self.inputSize, prod(self.inputSize), dqdx.shape))
        return dqdx, newDataArrangement
    
    def _backProp(self, dqda, data, dataArrangement, bpHint):
        '''Default pass through version. Override in derived classes.'''
        raise Exception('Implement in derived class')
    
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

    def initialize(self, trainParams = None, seed = None):
        if self.isInitialized:
            raise Exception('Layer was already initialized')
        self._initialize(trainParams = trainParams, seed = seed)
        self.isInitialized = True

    def _initialize(self, trainParams = None, seed = None):
        '''Default no-op version. Override in derived class.'''
        pass

    def train(self, data, dataArrangement, trainParams = None, quick = False):
        if self.isTrained:
            raise Exception('Layer was already trained')
        self._checkDataArrangement(data, dataArrangement)
        #ticw = time.time()
        #ticc = time.clock()
        self._train(data, dataArrangement, trainParams, quick)
        #print 'Layer took %.3fs wall time to train (%.3fs cpu time)' % (time.time() - ticw, time.clock()-ticc)
        self.isTrained = True

    def _train(self, data, dataArrangement, trainParams = None, quick = False):
        '''Default no-op version. Override in derived class.'''
        pass

    def plot(self, data, dataArrangement, saveDir = None, prefix = None):
        if not self.isTrained:
            raise Exception('Layer not trained yet')
        self._checkDataArrangement(data, dataArrangement)
        self._plot(data, dataArrangement, saveDir, prefix)

    def _plot(self, data, dataArrangement, saveDir = None, prefix = None):
        '''Default no-op version. Override in derived class. It is up to the layer
        what to plot.
        '''
        pass
        
        



######################
# Data
######################

class DataLayer(Layer):

    isDataLayer = True

    def __init__(self, params):
        super(DataLayer, self).__init__(params)

    def calculateSeesPatches(self):
        raise Exception('must implement in derived class')

    def getOutputSize(self):
        raise Exception('must implement in derived class')

    def getData(self):
        raise Exception('must implement in derived class')



class DummyDataLayer(DataLayer):
    '''For when you just need to specify a data layer. Only requires an output size.'''
    
    def __init__(self, params):
        super(DummyDataLayer, self).__init__(params)
        self.outputSize = params['outputSize']
        self.stride = (1,)   # Kind of fake...

    def calculateSeesPatches(self):
        '''For completeness. Always returns (1,)'''
        return (1,)

    def getOutputSize(self):
        return self.outputSize



class ImageDataLayer(DataLayer):

    def __init__(self, params):
        super(ImageDataLayer, self).__init__(params)

        self.imageSize = params['imageSize']
        self.patchSize = params['patchSize']
        self.stride = params['stride']

        # Only 2D data types for now (color is fine though)
        assert len(self.imageSize) == 2
        assert len(self.patchSize) == 2
        assert len(self.stride) == 2
        assert len(self.imageSize) == len(self.stride), 'imageSize and stride must be same length'

    def calculateSeesPatches(self):
        '''For completeness. Always returns (1, 1, 1...)'''
        return (1,) * len(self.stride)

    def numPatches(self):
        '''How many patches fit within the data. Rounds down.'''
        return tuple([(ims-ps)/st+1 for ims,ps,st in zip(self.imageSize, self.patchSize, self.stride)])



class UpsonData3(ImageDataLayer):

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



class NYU2_Labeled(ImageDataLayer):

    def __init__(self, params):
        '''
        Loads this dataset: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
        '''
        super(NYU2_Labeled, self).__init__(params)

        assert self.imageSize == (480,640)

        self.colorChannels = params['colorChannels']
        self.depthChannels = params['depthChannels']
        self.channels = self.colorChannels + self.depthChannels
        
        assert self.colorChannels in (0,1,3)
        assert self.depthChannels in (0,1)
        assert self.channels > 0

    def getOutputSize(self):
        if self.channels == 1:
            return self.patchSize
        else:
            return (self.patchSize[0], self.patchSize[1], self.channels)

    def getData(self, patchSize, number, seed = None):
        patches, labels = self.getDataAndLabels(patchSize, number, seed)
        return patches

    def getDataAndLabels(self, patchSize, number, seed = None):
        approxMbCachefile = prod(patchSize) * number * 4.0 / 1000000
        if approxMbCachefile < MAX_CACHE_SIZE_MB:
            print 'NYU2_Labeled layer:: Predicted approx size of %f MB, using cache' % (approxMbCachefile)
            patches, labels = cached(loadNYU2Data, patchSize = patchSize, number = number,
                                     rgbColors = self.colorChannels, depthChannels = self.depthChannels,
                                     seed = seed)
        else:
            # Skip cache, just run
            print 'NYU2_Labeled layer:: Skipping cache, approx size of %f MB > max cache size of %s MB' % (approxMbCachefile, repr(MAX_CACHE_SIZE_MB))
            patches, labels = loadNYU2Data(patchSize = patchSize, number = number,
                                           rgbColors = self.colorChannels, depthChannels = self.depthChannels,
                                           seed = seed)
        return patches, labels



class CS294Images(ImageDataLayer):
    '''
    Loads pre-whitened patches from this dataset: http://www.stanford.edu/class/cs294a/handouts.html
    Whitened range roughly -2.03 to 2.86.
    '''

    def __init__(self, params):
        super(CS294Images, self).__init__(params)

        self.colors = params['colors']

        assert self.imageSize == (512,512)
        assert self.colors == 1

    def getOutputSize(self):
        return self.patchSize

    def getData(self, patchSize, number, seed = None):
        patches = cached(loadCS294Images, patchSize = patchSize, number = number, seed = seed)

        return patches



######################
# Whitening and other normalization
######################

class NormalizingLayer(TrainableLayer):

    def __init__(self, params):
        super(NormalizingLayer, self).__init__(params)



class PCAWhiteningLayer(NormalizingLayer):

    def __init__(self, params):
        super(PCAWhiteningLayer, self).__init__(params)
        self.pcaWhiteningDataNormalizer = None

    def _train(self, data, dataArrangement, trainParams = None, quick = False):
        self.pcaWhiteningDataNormalizer = PCAWhiteningDataNormalizer(data)

    def _forwardProp(self, data, dataArrangement, sublayer):
        dataWhite, junk = self.pcaWhiteningDataNormalizer.raw2normalized(data, unitNorm = True)
        return dataWhite, dataArrangement



class ScaleClipLayer(NormalizingLayer):
    '''Subtracts mean of each example (not each dimension), clips to some
    multiple of S, where S is the the overall standard devation, then
    scales to the given [min, max] range. Same normalization as in CS294
    (see last few lines of loadCS294Images() function).
    '''

    def __init__(self, params):
        super(ScaleClipLayer, self).__init__(params)
        self.minVal  = float(params['min'])
        self.maxVal  = float(params['max'])
        self.clipStd = float(params['clipStd'])
        assert self.maxVal > self.minVal
        assert self.clipStd > 0

        self.thresh = None

    def _train(self, data, dataArrangement, trainParams = None, quick = False):
        dataNormed = data - data.mean(0)
        self.thresh = dataNormed.std() * self.clipStd

    def _forwardProp(self, data, dataArrangement, sublayer):
        dataNormed = data - data.mean(0)
        dataNormed = maximum(minimum(dataNormed, self.thresh), -self.thresh) / self.thresh   # scale to -1 to 1
        dataNormed = (dataNormed + 1) * ((self.maxVal-self.minVal)/2.0) + self.minVal        # rescale to minVal to maxVal
        return dataNormed, dataArrangement



######################
# Stretch
######################

class StretchingLayer(TrainableLayer):
    '''Stretches the data values along each dimension to have a given min/max'''

    def __init__(self, params):
        super(StretchingLayer, self).__init__(params)
        self.minVal = float(params['min'])
        self.maxVal = float(params['max'])

    def _train(self, data, dataArrangement, trainParams = None, quick = False):
        self.dataMin = data.min(1)
        self.dataMax = data.max(1)

    def _forwardProp(self, data, dataArrangement, sublayer):
        '''ret = mm * data + bb'''
        epsilon = 1e-8  # for dimensions with 0 variance
        mm = (self.maxVal - self.minVal) / (self.dataMax - self.dataMin + epsilon)
        bb = self.minVal - mm * self.dataMin

        scaledData = (data.T * mm + bb).T

        return scaledData, dataArrangement



######################
# Learning
######################

class TicaLayer(TrainableLayer):

    nSublayers = 2   # hidden representation + pooled representation

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

    def _initialize(self, trainParams = None, seed = None):
        # Set up untrained TICA
        self.tica = TICA(nInputs            = prod(self.inputSize),
                         hiddenLayerShape   = self.hiddenSize,
                         neighborhoodParams = self.neighborhood,
                         lambd              = self.lambd,
                         epsilon            = self.epsilon,
                         initWW             = False)
        self.tica.initWW(seed)

    def _train(self, data, dataArrangement, trainParams, quick = False):
        maxFuncCalls = trainParams['maxFuncCalls']
        if quick:
            print 'QUICK MODE: chopping maxFuncCalls from %d to 1!' % maxFuncCalls
            maxFuncCalls = 1
            

        tic = time.time()
        self.tica.learn(data, maxFun = maxFuncCalls)
        execTime = time.time() - tic
        #if logDir:
        #    saveToFile(os.path.join(logDir, (prefix if prefix else '') + 'tica.pkl.gz'), tica)    # save learned model

        beginTotalCost, beginPoolingCost, beginReconstructionCost = self.tica.costLog[0]
        endTotalCost,   endPoolingCost,   endReconstructionCost   = self.tica.costLog[-1]

        print 'Training stats:'
        print '  begin {totalCost, poolingCost, reconCost} = %f, %f, %f' % (beginTotalCost, beginPoolingCost, beginReconstructionCost)
        print '    end {totalCost, poolingCost, reconCost} = %f, %f, %f' % (endTotalCost, endPoolingCost, endReconstructionCost)
        print '  Number of cost evals:', len(self.tica.costLog)
        print '  Training time (wall)', execTime

        # Plot some results
        #plotImageRicaWW(tica.WW, imgShape, saveDir, tileShape = hiddenLayerShape, prefix = pc('WW_iterFinal'))

    def _forwardProp(self, data, dataArrangement, sublayer):
        hidden, absPooledActivations = self.tica.getRepresentation(data)
        
        if sublayer == 0:
            return hidden, dataArrangement
        elif sublayer == 1:
            return absPooledActivations, dataArrangement
        else:
            raise Exception('Unknown sublayer: %s' % sublayer)
    
    def _plot(self, data, dataArrangement, saveDir = None, prefix = None):
        '''Default no-op version. Override in derived class. It is up to the layer
        what to plot.
        '''
        if saveDir:
            self.tica.plotCostLog(saveDir, prefix)



class SparseAELayer(TrainableLayer):

    ##nSublayers = 2   # hidden representation + pooled representation

    def __init__(self, params):
        super(SparseAELayer, self).__init__(params)

        self.hiddenSize = params['hiddenSize']
        self.beta       = params['beta']
        self.rho        = params['rho']
        self.lambd      = params['lambd']

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W1shape = None
        self.b1shape = None
        self.W2shape = None
        self.b2shape = None

        self._costLog = []    # this version is a list. The property costLog returns a numpy array.

        assert not isinstance(self.hiddenSize, tuple)  # should just be a number

    def _calculateOutputSize(self, inputSize):
        return (self.hiddenSize, )    # return as tuple

    def _initialize(self, trainParams = None, seed = None):
        # Initialize weights

        initb1       = trainParams['initb1']
        initW2asW1_T = trainParams['initW2asW1_T']
        assert initb1 in ('zero', 'approx')
        assert initW2asW1_T in (True, False)
        
        rng = random.RandomState(seed)      # if seed is None, this takes its seed from timer

        self.W1shape = (self.hiddenSize, prod(self.inputSize))
        self.b1shape = (self.hiddenSize,)
        self.W2shape = (prod(self.inputSize), self.hiddenSize)
        self.b2shape = (prod(self.inputSize),)

        radius = sqrt(6.0 / (prod(self.inputSize) + prod(self.outputSize) + 1))
        self.W1 = rng.uniform(-radius, radius, self.W1shape)
        if initb1 == 'zero':
            self.b1 = zeros(self.b1shape)
        else:  # 'approx'
            self.b1 = invSigmoid01(self.rho) * ones(self.b1shape)
        if initW2asW1_T:
            self.W2 = self.W1.T.copy()
        else:
            self.W2 = rng.uniform(-radius, radius, self.W2shape)
        self.b2 = zeros(self.b2shape)

    def _train(self, data, dataArrangement, trainParams, quick = False):
        #pdb.set_trace()
        
        maxFuncCalls = trainParams['maxFuncCalls']
        if quick:
            print 'QUICK MODE: chopping maxFuncCalls from %d to 1!' % maxFuncCalls
            maxFuncCalls = 1

        tic = time.time()

        theta0 = concatenate((self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()))

        results = minimize(self._costAndLog,
                           theta0,
                           (data, self.hiddenSize, self.beta, self.rho, self.lambd),
                           jac = True,    # cost function retuns both value and gradient
                           method = 'L-BFGS-B',
                           options = {'maxiter': maxFuncCalls, 'disp': True})

        fval = results['fun']
        wallSeconds = time.time() - tic
        print 'Optimization results:'
        for key in ['status', 'nfev', 'success', 'fun', 'message']:
            print '  %20s: %s' % (key, results[key])
        print '  %20s: %s' % ('fval', fval)
        print '  %20s: %s' % ('fval/example', fval/data.shape[1])
        print '  %20s: %s' % ('wall time', fmtSeconds(wallSeconds))
        print '  %20s: %s' % ('wall time/funcall', fmtSeconds(wallSeconds / results['nfev']))

        theta = results['x']

        # Unpack theta into W and b parameters
        begin = 0
        self.W1 = reshape(theta[begin:begin+prod(self.W1shape)], self.W1shape);    begin += prod(self.W1shape)
        self.b1 = reshape(theta[begin:begin+prod(self.b1shape)], self.b1shape);    begin += prod(self.b1shape)
        self.W2 = reshape(theta[begin:begin+prod(self.W2shape)], self.W2shape);    begin += prod(self.W2shape)
        self.b2 = reshape(theta[begin:begin+prod(self.b2shape)], self.b2shape);

    def _costAndLog(self, theta, data, hiddenSize, beta, rho, lambd):
        #if len(self.costLog) in (0, 100):
        #    print 'At iteration %d, stopping' % len(self.costLog)
        #    pdb.set_trace()
        output = autoencoderCost(theta, data, hiddenSize, beta, rho, lambd, output = 'sepcosts')
        cost, grad, reconCost, sparseCost, weightCost = output

        self._costLog.append([cost, reconCost, sparseCost, weightCost])
        print 'f =', cost, '|grad| =', linalg.norm(grad)

        return cost, grad
        
    def _forwardProp(self, data, dataArrangement, sublayer, outputBpHint = False):
        if sublayer != 0:
            raise Exception('Unknown sublayer: %s' % sublayer)

        output = autoencoderForwardprop(self.W1, self.b1, data, outputSigmoidDeriv = outputBpHint)
        if outputBpHint:
            a2, sigmoidDeriv = output
            bpHint = (sigmoidDeriv,)
            return a2, dataArrangement, bpHint
        else:
            a2 = output
            return a2, dataArrangement

    def _backProp(self, dqda, data, dataArrangement, bpHint = None):
        sigmoidDeriv = bpHint[0] if bpHint else None
        dqdx = autoencoderBackprop(self.W1, self.b1, dqda, data, sigmoidDeriv = sigmoidDeriv)
        return dqdx, dataArrangement
        
    @property
    def costLog(self):
        return array(self._costLog)
    
    def _plot(self, data, dataArrangement, saveDir = None, prefix = None):
        if prefix is None:
            prefix = ''
        if saveDir:
            # plot sparsity/reconstruction costs over time
            costs = self.costLog    # costs is [cost, reconCost, sparseCost, weightCost]
            pyplot.figure()
            pyplot.hold(True)
            pyplot.plot(costs[:,1], 'r-', costs[:,2], 'b-', costs[:,3], 'g-', costs[:,0], 'k-')
            pyplot.xlabel('iteration'); pyplot.ylabel('cost')
            pyplot.legend(('recon', 'sparsity', 'weight', 'total'))
            pyplot.savefig(os.path.join(saveDir, prefix + 'cost.png'))
            pyplot.savefig(os.path.join(saveDir, prefix + 'cost.pdf'))

            pyplot.ylim((pyplot.ylim()[0], percentile(costs[:,0], 90)))   # rescale to show only the smallest 90% of the data
            pyplot.savefig(os.path.join(saveDir, prefix + 'cost_zoom.png'))
            pyplot.savefig(os.path.join(saveDir, prefix + 'cost_zoom.pdf'))

            # plot (in, weight, rep, weight, out) histograms
            theta = concatenate((self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()))
            output = autoencoderCost(theta, data, self.hiddenSize, self.beta, self.rho, self.lambd, output='full')
            cost, grad, reconCost, sparseCost, weightCost, z2, a2, z3, a3, rho_hat = output
            pyplot.clf()
            pyplot.subplot(5,1,1); pyplot.hist(data.flatten(), bins=30)
            pyplot.subplot(5,2,3); pyplot.hist(self.W1.flatten(), bins=30)
            pyplot.subplot(5,2,4); pyplot.hist(self.b1.flatten(), bins=30)
            pyplot.subplot(5,1,3); pyplot.hist(a2.flatten(), bins=30)
            pyplot.subplot(5,2,7); pyplot.hist(self.W2.flatten(), bins=30)
            pyplot.subplot(5,2,8); pyplot.hist(self.b2.flatten(), bins=30)
            pyplot.subplot(5,1,5); pyplot.hist(a3.flatten(), bins=30)

            pyplot.savefig(os.path.join(saveDir, prefix + 'ae_hists.png'))
            pyplot.savefig(os.path.join(saveDir, prefix + 'ae_hists.pdf'))
                                    
            if len(self.inputSize) > 1:
                # Plot W1 and W2
                imgShape = self.inputSize
                tileSideLength = int(ceil(sqrt(self.hiddenSize)))
                tileShape = (tileSideLength, tileSideLength)
                plotImageData(self.W1.T, imgShape, saveDir=saveDir, prefix=prefix + 'direct_W1', tileShape=tileShape, onlyRescaled=True)
                plotImageData(self.W2,   imgShape, saveDir=saveDir, prefix=prefix + 'direct_W2', tileShape=tileShape, onlyRescaled=True)

                # Plot 
                rows = 12
                cols = 1
                width = 4    # how many spaces for each window
                pyplot.figure(figsize = (cols*width*2,rows*2))
                for ii in range(rows * cols):
                    start = width*ii
                    pyplot.subplot(rows, width*cols, start+1)
                    pyplot.imshow(reshape(data[:,ii], self.inputSize), vmin=0, vmax=1, cmap = cm.Greys_r, interpolation = 'nearest')
                    pyplot.subplot(rows, width*cols, start+2)
                    pyplot.hold(True)
                    pyplot.vlines(arange(a2.shape[0]), 0, a2[:,ii], 'b')
                    pyplot.plot(a2[:,ii], 'bo')
                    pyplot.subplot(rows, width*cols, start+3)
                    pyplot.imshow(reshape(a3[:,ii], self.inputSize), vmin=0, vmax=1, cmap = cm.Greys_r, interpolation = 'nearest')
                    st = 'R: %g' % (.5 * ((a3[:,ii] - data[:,ii])**2).sum())
                    pyplot.text(self.inputSize[1], 1, st)
                pyplot.savefig(os.path.join(saveDir, prefix + 'recon.png'))
                pyplot.savefig(os.path.join(saveDir, prefix + 'recon.pdf'))
            else:
                # For now: skip plotting the W1 and W2 for layers with non-visualizable inputs
                pass
                #imgSizeLength = int(ceil(sqrt(self.inputSize)))
                #imgShape = (imgSizeLength, imgSizeLength)
                
            pyplot.close()



class Fro1Layer(TrainableLayer):

    def __init__(self, params):
        super(Fro1Layer, self).__init__(params)

        self._costLog = []    # this version is a list. The property costLog returns a numpy array.

    def _calculateOutputSize(self, inputSize):
        return (self.hiddenSize, )    # return as tuple

    def _initialize(self, trainParams = None, seed = None):
        # Initialize weights
        pass

    def _train(self, data, dataArrangement, trainParams, quick = False):
        #pdb.set_trace()
        pass

    def _costAndLog(self, theta, data, hiddenSize, beta, rho, lambd):
        return cost, grad
        
    def _forwardProp(self, data, dataArrangement, sublayer):
        pass
        
    @property
    def costLog(self):
        return array(self._costLog)
    
    def _plot(self, data, dataArrangement, saveDir = None, prefix = None):
        print 'TODO: plot'



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

    def _forwardProp(self, data, dataArrangement, sublayer):
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

    def _forwardProp(self, data, dataArrangement, sublayer):
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

    def _forwardProp(self, data, dataArrangement, sublayer):
        '''This is where the concatenation actually takes place.'''

        newLayerShape = tuple([1+(sliceShape - concat) / stride for sliceShape,concat,stride
                               in zip(dataArrangement.sliceShape, self.concat, self.stride)])
        remainders    = tuple([(sliceShape - concat) % stride for sliceShape,concat,stride
                               in zip(dataArrangement.sliceShape, self.concat, self.stride)])
        tooSmall = sum([nls < 1 for nls in newLayerShape])  # must be at least 1 in every dimension
        if tooSmall or any(remainders):
            raise Exception('%s cannot be evenly concatenated by concat %s and stride %s'
                            % (dataArrangement, self.concat, self.stride))

        reshapedData = reshape(data, (data.shape[0], dataArrangement.nSlices,) + dataArrangement.sliceShape)

        newDataArrangement = DataArrangement(newLayerShape, dataArrangement.nSlices)

        # BEGIN: 2D data assumption
        # Note: this only works for 2D data! Add other cases if needed.

        assert len(self.concat) == 2, 'Only works for 2D data for now!'
        assert len(dataArrangement.sliceShape) == 2, 'Only works for 2D data for now!'

        concatenatedData = zeros((prod(self.outputSize),
                                  prod(newLayerShape) * dataArrangement.nSlices))
        
        oldPatchLength = prod(self.inputSize)
        newPatchCounter = 0
        # Note: this code assumes the default numpy flattening order:
        # C-order, or row-major order.
        for layerIdx in xrange(dataArrangement.nSlices):
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



layerClassNames = {'data':           'dataClass',       # load the class specified by DataClass
                   'whitener':       'whitenerClass',   # load the class specified by whitenerClass
                   'scaleclip':       ScaleClipLayer,
                   'stretch':         StretchingLayer,
                   'tica':            TicaLayer,
                   'ae':              SparseAELayer,
                   'fro1':            Fro1Layer,
                   'downsample':      DownsampleLayer,
                   'lcn':             LcnLayer,
                   'concat':          ConcatenationLayer,
                   }







































def check_AE_backprop():
    random.seed(0)

    dimInput = 3
    dimRep   = 2
    numExamples = 5

    layer = SparseAELayer({'name':         'ae1',
                           'type':         'ae',
                           'hiddenSize':   dimRep,
                           'beta':         3.0,
                           'rho':          .01,
                           'lambd':        .0001,
                           })

    # random data
    XX = random.normal(.5, .25, (dimInput, numExamples))

    tp = {}
    tp['ae1'] = {'examples': 0,
                'initb1': 'zero',        # 'zero' for 0s, or 'approx' for an approx value to start at avg activation of rho
                'initW2asW1_T': False,
                'method': 'lbfgs',
                'maxFuncCalls': 300,
             }
    tp['ae2'] = tp['ae1']

    sl.train(tp, onlyInit = True)

    # Make biases non-zero for more generality
    sl.layers[1].b1 = random.normal(0, .1, sl.layers[1].b1.shape)
    sl.layers[1].b2 = random.normal(0, .1, sl.layers[1].b2.shape)
    sl.layers[2].b1 = random.normal(0, .1, sl.layers[2].b1.shape)
    sl.layers[2].b2 = random.normal(0, .1, sl.layers[2].b2.shape)
    
    #sl.forwardProp(XX, DataArrangement((1,), numExamples))

    singleExampleArrangement = DataArrangement((1,), 1)

    repUnit = 1  # which highest level unit to consider
    activationFunction = lambda xx : float(sl.forwardProp(xx, singleExampleArrangement, quiet = True)[0][repUnit])

    xx = XX[:,0]

    print 'sl.forwardProp: act is', activationFunction(xx)

    def actAndGrad(xx):
        reps = [xx]
        # Forward prop
        for ii in range(2):
            rep,newDA = sl.layers[ii+1].forwardProp(reps[ii], singleExampleArrangement, quiet = True)
            reps.append(rep)

        # Back prop to calculate gradient
        dqda_top = zeros(dimRep2)
        dqda_top[repUnit] = 1
        dqdas = [None, None, dqda_top]
        for ii in reversed(range(2)):
            dqda,newDA = sl.layers[ii+1].backProp(dqdas[ii+1], reps[ii], singleExampleArrangement, quiet = True)
            dqdas[ii] = dqda
        return float(reps[-1][repUnit]), dqdas[0].flatten()

    print 'manual forwardProp: act is', actAndGrad(xx)[0]
    
    numericalCheckVectorGrad(actAndGrad, xx)



def tests():
    check_AE_backprop()



if __name__ == '__main__':
    tests()

