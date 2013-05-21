#! /usr/bin/env python

import pdb
import os
import gc
from numpy import zeros, prod, reshape
from IPython import embed

from util.dataLoaders import loadFromPklGz, saveToFile
from util.misc import dictPrettyPrint, relhack, Tic
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, NYU2_Labeled, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from visualize import plotImageData, plotTopActivations



class StackedLayers(object):

    def __init__(self, layerList):

        self.layers = []

        # 0. Check for duplicate names
        self.layerNames = [layer['name'] for layer in layerList]
        dups = set([x for x in self.layerNames if self.layerNames.count(x) > 1])
        if len(dups) > 0:
            raise Exception('Duplicate layer names: %s' % dups)

        # 1. Construct all layers
        for ii, layerDict in enumerate(layerList):
            print 'constructing layer %d:' % ii
            dictPrettyPrint(layerDict, prefix = '  ')

            layerClass = layerClassNames[layerDict['type']]
            if isinstance(layerClass, basestring):
                actualLayerClass = eval(layerDict[layerClass])
                layer = actualLayerClass(layerDict)
            else:
                layer = layerClass(layerDict)

            # Checks
            if ii == 0 and not layer.isDataLayer:
                raise Exception('First layer must be data layer, but it is %s' % repr(layer))
            if ii != 0 and layer.isDataLayer:
                raise Exception('Only the first layer can be a data layer, but layer %d is %s' % (ii, repr(layer)))

            # Populate inputSize, outputSize, seesPatches, and distToNeighbor
            if ii == 0:
                # For data layer
                layer.outputSize = layer.getOutputSize()
                layer.seesPatches = layer.calculateSeesPatches()  # no input needed for DataLayer
            else:
                prevLayer = self.layers[ii-1]

                # only for non-data layers
                layer.inputSize    = prevLayer.outputSize
                layer.outputSize   = layer.calculateOutputSize(layer.inputSize)
                if prevLayer.isDataLayer:
                    prevLayerSees      = (1,) * len(prevLayer.stride)
                    prevDistToNeighbor = (1,) * len(prevLayer.stride)
                else:
                    prevLayerSees      = prevLayer.seesPatches
                    prevDistToNeighbor = prevLayer.distToNeighbor
                layer.seesPatches    = layer.calculateSeesPatches(prevLayerSees, prevDistToNeighbor)
                layer.distToNeighbor = layer.calculateDistToNeighbor(prevDistToNeighbor)

            self.layers.append(layer)

        # 2. Post construction checks...
        for ii, layer in enumerate(self.layers):
            pass
        print 'layers look good (purely cursory check)'


    def printStatus(self):
        dataLayer = self.layers[0]
        for ii, layer in reversed(list(enumerate(self.layers))):
            print 'layer %2d: %-20s' % (ii, '%s (%s)' % (layer.name, layer.layerType)),
            if ii == 0:
                st = 'outputs size %s' % repr(layer.outputSize),
            else:
                st = 'maps size %s -> %s' % (repr(layer.inputSize), repr(layer.outputSize)),
            print '%-32s' % st,

            try:
                st = 'sees %s' % repr(self._seesPixels(layer, dataLayer))
            except AttributeError:
                st = '--'
            print '%-16s' % st,
            if layer.trainable:
                print 'trained' if layer.isTrained else 'not trained'
            else:
                print '--'

    def forwardProp(self, data, dataArrangement, layerIdx = None):
        '''Push the given data through the layers 0, 1, ..., layerIdx.
        If layerIdx is None, set layerIdx = max
        '''

        if layerIdx is None: layerIdx = len(self.layers)-1

        print 'forwardProp from layer 1 through %d' % (layerIdx)
        currentRep = data
        currentArrangement = dataArrangement
        for ii in range(1, layerIdx+1):
            layer = self.layers[ii]
            print '  fp through layer %d - %s (%s)' % (ii, layer.name, layer.layerType)
            newRep, newArrangement = layer.forwardProp(currentRep, currentArrangement)
            print ('    layer transformed data from\n      %s, %s ->\n      %s, %s'
                   % (currentRep.shape, currentArrangement, newRep.shape, newArrangement))
            currentRep = newRep
            currentArrangement = newArrangement
        return currentRep, currentArrangement

    def train(self, trainParams, saveDir = None, quick = False):
        # check to make sure each trainParam matches a known layer...
        for layerName in trainParams.keys():
            if layerName not in self.layerNames:
                raise Exception('unknown layer name in param file: %s' % layerName)
        # ...and each trainable but untrained layer has a trainParam present
        for layerIdx, layer in enumerate(self.layers):
            if layer.trainable and not layer.isTrained:
                if not layer.name in trainParams:
                    raise Exception('Param file missing training params for layer: %s' % layer.name)

        dataLayer = self.layers[0]

        trainedSomething = False
        for layerIdx, layer in enumerate(self.layers):
            if layer.trainable and not layer.isTrained:
                trainedSomething = True
                print '\n' + '*' * 40
                print 'training layer %d - %s (%s)' % (layerIdx, layer.name, layer.layerType)
                print '*' * 40 + '\n'

                layer.initialize()

                layerTrainParams = trainParams[layer.name]
                numExamples = layerTrainParams['examples']
                if quick:
                    numExamples = 1000
                    print 'QUICK MODE: chopping training examples to 1000!'

                # Make sure layer.sees, data.stride, and data.patchSize are all same len
                assert len(layer.seesPatches) == len(dataLayer.patchSize)
                assert len(layer.seesPatches) == len(dataLayer.stride)

                # Get data
                print 'gc.collect found', gc.collect(), 'objects'
                trainRawDataLargePatches, trainRawDataPatches = self.getDataForLayer(layerIdx, numExamples)
                print 'Memory used to store trainRawDataPatches: %g MB' % (trainRawDataPatches.nbytes/1e6)

                # Push data through N-1 layers
                dataArrangementLayer0 = DataArrangement(layerShape = layer.seesPatches, nLayers = numExamples)
                tic = Tic('forward prop')
                trainPrevLayerData, dataArrangementPrevLayer = self.forwardProp(trainRawDataPatches, dataArrangementLayer0, layerIdx-1)
                tic()
                print 'Memory used to store trainPrevLayerData: %g MB' % (trainPrevLayerData.nbytes/1e6)

                # Free the raw patches from memory
                del trainRawDataLargePatches
                del trainRawDataPatches
                print 'gc.collect found', gc.collect(), 'objects'

                # Train layer
                tic = Tic('train')
                layer.train(trainPrevLayerData, dataArrangementPrevLayer, layerTrainParams, quick = quick)
                tic()
                print 'training done for layer %d - %s (%s)' % (layerIdx, layer.name, layer.layerType)
                
                # Checkpoint and plot
                if saveDir:
                    fileFinal = os.path.join(saveDir, 'stackedLayers.pkl.gz')
                    fileTmp = fileFinal + '.tmp'
                    print 'Saving checkpoint...'
                    saveToFile(fileTmp, self)
                    os.rename(fileTmp, fileFinal)
                    print 'Saving these StackedLayers...'
                    self.printStatus()
                    print '... to %s' % fileFinal

                    tic = Tic('plot')
                    prefix = 'layer_%02d_%s_' % (layerIdx, layer.name)
                    layer.plot(trainPrevLayerData, dataArrangementPrevLayer, saveDir, prefix)
                    tic()


        if not trainedSomething:
            print '\nNothing to train. Maybe it was already finished?'

    def _seesPixels(self, layer, dataLayer):
        seesPixels = tuple([ps + st * (sp-1) for sp,ps,st in zip(layer.seesPatches,dataLayer.patchSize, dataLayer.stride)])
        return seesPixels

    def getDataForLayer(self, layerIdx, numExamples):
        layer = self.layers[layerIdx]
        dataLayer = self.layers[0]
        
        # How many pixels this layer sees at once
        seesPixels = self._seesPixels(layer, dataLayer)

        tic = Tic('get patches')
        rawDataLargePatches = dataLayer.getData(seesPixels, numExamples, seed = 0)
        tic()

        # Reshape into patches
        rawDataPatches = zeros((prod(dataLayer.patchSize), prod(layer.seesPatches)*numExamples))

        # BEGIN: 2D data assumption
        # Note: this only works for 2D data! Add other cases if needed.
        assert len(layer.seesPatches) == 2, 'Only works for 2D data for now!'
        counter = 0
        sp0,sp1 = layer.seesPatches
        ps0,ps1 = dataLayer.patchSize
        st0,st1 = dataLayer.stride
        for largePatchIdx in xrange(rawDataLargePatches.shape[1]):
            thisLargePatch = reshape(rawDataLargePatches[:,largePatchIdx], seesPixels)
            # Note: this code flattens in the default numpy
            # flattening order: C-order, or row-major order.
            for ii in xrange(sp0):
                for jj in xrange(sp1):
                    rawDataPatches[:,counter] = thisLargePatch[(ii*st0):(ii*st0+ps0),(jj*st1):(jj*st1+ps1)].flatten()
                    counter += 1
        # just check last patch 
        assert rawDataPatches.shape[0] == len(thisLargePatch[(ii*st0):(ii*st0+ps0),(jj*st1):(jj*st1+ps1)].flatten())
        assert rawDataPatches.shape[1] == counter
        # END: 2D data assumption
        #################

        return rawDataLargePatches, rawDataPatches
    
    def visLayer(self, layerIdx, saveDir = None):
        layer     = self.layers[layerIdx]
        dataLayer = self.layers[0]

        numExamples = 100000
        prefix = 'layer_%02d_%s_' % (layerIdx, layer.name)
        seesPixels = self._seesPixels(layer, dataLayer)

        # Get data and forward prop
        rawDataLargePatches, rawDataPatches = self.getDataForLayer(layerIdx, numExamples)
        dataArrangementLayer0 = DataArrangement(layerShape = layer.seesPatches, nLayers = numExamples)
        tic = Tic('forward prop')
        activations, dataArrangement = self.forwardProp(rawDataPatches, dataArrangementLayer0, layerIdx)
        tic()
        
        # 0. Inputs
        plotImageData(rawDataLargePatches, seesPixels, saveDir, prefix = prefix + '0_input', tileShape = (10,10))

        # 1. Top activations
        plotTopActivations(activations, rawDataLargePatches, seesPixels, saveDir = saveDir,
                           nActivations = 50, nSamples = 20, prefix = prefix + '1_topact')

    def visAllTica(self, layerIdx, saveDir = None):
        # NOTE: untested...
        for layerIdx, layer in enumerate(self.layers):
            if layer.type == 'tica' and layer.isTrained:
                print 'vis layer %2d: %s' % (layerIdx, layer.name)
                self.visLayer(layerIdx, saveDir)
