#! /usr/bin/env ipython

'''
Research code

Jason Yosinski

Note: This coding was mostly done in a rush. Not that it is
necessarily horrible, but there may be duplicated functionality
between this file and others.
'''

import ipdb as pdb
import os, sys, time
from numpy import *
from PIL import Image
from GitResultsManager import resman
from IPython.parallel import Client

from tica import neighborMatrix, TICA
from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, loadUpsonData, loadUpsonData3, loadRandomData, saveToFile, loadFromPklGz
from util.cache import cached
from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
from util.misc import pt, pc
from makeData.makeUpsonRovio3 import randomSampleMatrixWithLabels, trainFilter, testFilter, exploreFilter



STACKED_NEIGHBORHOOD_SIZE = 2.0

class StackedTICA(object):
    '''Contains one or more TICA objects'''

    def __init__(self, layer1Tica, layer1Whitener, layer1DataPath, layerSizePlan, isColor, saveDir):
        ''''''
        self.l1dataPath = layer1DataPath      # path to image directories
        self.l1whitener = layer1Whitener
        self.isColor    = isColor
        self.ticas      = [layer1Tica]        # this fills up as training completes
        self.layerSizePlan = layerSizePlan    # this should be complete
        self.saveDir    = saveDir


    def learnNextLayer(self, params):
        nLayers = len(self.ticas)
        print 'StackedTica currently has %d layers, learning next' % nLayers

        # TODO: only works for one extra layer!
        Nw = 10
        Nwshift = 5
        Nsamples = 50000
        #Nsamples = 1000; print 'HACK!'

        # Get data and norm it
        nextLayerData = cached(makeNewData, self.ticas[-1], self.l1whitener, seed = 0,
                               isColor = self.isColor, Nw = Nw, Nwshift = Nwshift,
                               Nsamples = Nsamples)
        colNorms = sqrt(sum(nextLayerData**2, 0) + (1e-8))
        nextLayerData = nextLayerData / colNorms

        # Parameters
        if params['dataCrop']:
            print '\nWARNING: Cropping data from %d examples to only %d for debug\n' % (nextLayerData.shape[1], params['dataCrop'])
            nextLayerData = nextLayerData[:,:params['dataCrop']]
        if params['hiddenISize'] != params['hiddenJSize']:
            raise Exception('hiddenISize and hiddenJSize must be the same')
        hiddenLayerShape = (params['hiddenISize'], params['hiddenJSize'])
        neighborhoodParams = ('gaussian', params['neighborhoodSize'], 0, 0)
        if self.saveDir:
            layerLogDir = os.path.join(self.saveDir, 'layer_%02d' % (nLayers+1))
            os.makedirs(layerLogDir)
        else:
            layerLogDir = ''

        # Print/plot data stats
        if layerLogDir:
            pseudoImgShape = (int(sqrt(nextLayerData.shape[0])), int(sqrt(nextLayerData.shape[0])))
            plotImageData(nextLayerData, pseudoImgShape, layerLogDir, pc('data_raw'))
        printDataStats(nextLayerData)

        # Learn model
        tica = TICA(nInputs            = nextLayerData.shape[0],
                    hiddenLayerShape   = hiddenLayerShape,
                    neighborhoodParams = neighborhoodParams,
                    lambd              = params['lambd'],
                    epsilon            = 1e-5,
                    saveDir            = layerLogDir)

        manuallySkipExpensivePart = False
        if not manuallySkipExpensivePart:
            beginTotalCost, beginPoolingCost, beginReconstructionCost, grad = tica.cost(tica.WW, nextLayerData)

            tic = time.time()
            tica.learn(nextLayerData, maxFun = params['maxFuncCalls'])
            execTime = time.time() - tic
            if layerLogDir:
                saveToFile(os.path.join(layerLogDir, 'tica.pkl.gz'), tica)    # save learned model

            endTotalCost, endPoolingCost, endReconstructionCost, grad = tica.cost(tica.WW, nextLayerData)

            print 'beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime ='
            print [beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime]
        else:
            pdb.set_trace()

        # Plot some results
        #plotImageRicaWW(tica.WW, imgShape, saveDir, tileShape = hiddenLayerShape, prefix = pc('WW_iterFinal'))
        if layerLogDir:
            self.plotResults(layerLogDir, tica, nextLayerData, pseudoImgShape, hiddenLayerShape)

        self.ticas.append(tica)


    def getRepresentation(self, data):
        '''Assumes data is one example per column. Returns the pooled
        activations. Figures out which layer to use based on the data
        size.'''

        Nwbig = int(sqrt(data.shape[0]))
        if Nwbig not in self.layerSizePlan:
            raise Exception('Nwbig = %d not in %s, data shape was %s' % (Nwbig, repr(self.layerSizePlan), repr(data.shape)))
        layerIdx = self.layerSizePlan.index(Nwbig)

        if layerIdx > 1:
            raise Exception('not supported yet...')
        elif layerIdx == 1:
            Nwsmall = self.layerSizePlan[layerIdx-1]
            Nwshift = Nwbig - Nwsmall
            Nsamples = data.shape[1]
            stackedSmall = large2StackedSmall(data.T, Nwsmall, Nwshift).T   # expects one per row
            hidden,pooled = self.getRepresentation(stackedSmall)

            # LCN and subsample...
            hiddenSize = int(sqrt(self.ticas[1].nHidden))           # e.g. 15x15
            downsampleSize = int(sqrt(self.ticas[1].nInputs)) / 2   # e.g. 8x8
            downsampleSizeSq = downsampleSize**2
            downsampled = zeros((downsampleSize**2, Nsamples*4))   # store in columns for use with neighbor matrix

            gaussNeighbors = neighborMatrix((downsampleSize,downsampleSize), STACKED_NEIGHBORHOOD_SIZE, gaussian=True)
            for smallIdx in xrange(4*Nsamples):
                poolEmbedded = reshape(pooled[:,smallIdx], (hiddenSize, hiddenSize))     # 15x15
                downsampled[:,smallIdx] = poolEmbedded[::2,::2].flatten()                # 8x8 -> 64 column

            # 2. LCN
            vv = downsampled - dot(gaussNeighbors, downsampled)
            sig = sqrt(dot(gaussNeighbors, vv**2))
            cc = .01     # ss = sorted(sig.flatten()); ss[len(ss)/10] = 0.026 in one test. So .01 seems about right.
            yy = vv / maximum(cc, sig)

            # 3. Stack together 4x (one example per COLUMN)
            nextLayerInput = zeros((downsampleSizeSq * 4, Nsamples))
            for smallIdx in xrange(4*Nsamples):
                col    = smallIdx / 4
                offset = smallIdx % 4
                nextLayerInput[(offset*downsampleSizeSq):((offset+1)*downsampleSizeSq),col] = yy[:,smallIdx]

            hidden,pooled = self.ticas[1].getRepresentation(nextLayerInput)

            return pooled

        else:
            # layerIdx == 0
            whiteData,junk = self.l1whitener.raw2normalized(data)
            return self.ticas[0].getRepresentation(whiteData)



    def plotResults(self, layerLogDir, tica, data, imgShape, hiddenLayerShape):
            plotRicaActivations(tica.WW, data, layerLogDir, prefix = pc('activations_iterFinal'))
            plotRicaReconstructions(tica, data, imgShape, layerLogDir,
                                    unwhitener = None,
                                    tileShape = hiddenLayerShape, prefix = pc('recon_iterFinal'),
                                    number = 20)
            plotRicaReconstructions(tica, data, imgShape, layerLogDir,
                                    unwhitener = None,
                                    tileShape = hiddenLayerShape, prefix = pc('recon_hl_iterFinal'),
                                    number = 20, onlyHilights = True,
                                    hilightCmap = 'hot')



def large2StackedSmall(largeMatrix, Nw, Nwshift):
    # Sample each corner of the 1.5x patch (e.g. 4 patches 10px on a
    # side). Color order is [ii_r ii_g ii_b ii+1_r ii+1_g ii+1_b ...]
    Nsamples = largeMatrix.shape[0]   # one sample per row
    stackedSmall = zeros((4*Nsamples, Nw*Nw))
    NwLarge = Nw + Nwshift
    counter = 0
    for largeIdx in xrange(Nsamples):
        largePatch = reshape(largeMatrix[largeIdx,:], (NwLarge, NwLarge))
        for ii in (0,1):
            for jj in (0,1):
                stackedSmall[counter,:] = largePatch[(ii*Nwshift):(ii*Nwshift+Nw), (jj*Nwshift):(jj*Nwshift+Nw)].flatten()
                counter += 1
    return stackedSmall



def getBigAndSmallerSamples(fileFilter, whitener, seed, isColor, Nw, Nwshift, Nsamples):
    # Sample 1.5x larger windows (e.g. 15 px on a side). This is for
    # patches that overlap half with their
    # neighbors. largeSamples.shape is (Nsamples, Nw15^2)
    NwLarge = Nw + Nwshift
    largeSampleMatrix, labelMatrix, labelStrings = cached(randomSampleMatrixWithLabels, fileFilter, seed, isColor, Nw = NwLarge, Nsamples = Nsamples)

    stackedSmall = large2StackedSmall(largeSampleMatrix, Nw, Nwshift)

    stackedSmallWhite,junk = whitener.raw2normalized(stackedSmall.T)
    # dataWhite.shape = (Nw^2, 4*Nsamples). Now with one example per COLUMN!

    return largeSampleMatrix.T, stackedSmall.T, stackedSmallWhite, labelMatrix.T, labelStrings



def makeNewData(tica, layer1Whitener, seed, isColor, Nw, Nwshift, Nsamples):
    '''
    Nw like 10 for 10x10 patches
    Nwshift like 5 for 10x10 -> 15x15 patches'''
    #resman.start('junk', diary = False)

    tic = time.time()
    
    #tica  = loadFromPklGz('results/130402_033310_44cc757_master_psearchTica_UP/00022_data/tica.pkl.gz')
    #myres = loadFromPklGz('results/130402_033310_44cc757_master_psearchTica_UP/00022_data/myresults.pkl.gz')
    # myres['params']['dataPath'] is '../data/upson_rovio_2/train_10_50000_1c.pkl.gz'
    #layer1Whitener = loadFromPklGz('../data/upson_rovio_2/white/train_10_50000_1c.whitener.pkl.gz')

    #seed = seed
    #isColor = isColor

    nColors = 3 if isColor else 1

    if Nw%2 != 0:
        raise Exception('only works for even patch sizes')
    if isColor:
        raise Exception('no colors yet....')
    hiddenISize, hiddenJSize = tica.hiddenLayerShape
    if hiddenISize != hiddenJSize:
        raise Exception('representation embedding must be square (for now)')
    hiddenSize = hiddenISize

    temp = getBigAndSmallerSamples(trainFilter, layer1Whitener, seed, isColor, Nw, Nwshift, Nsamples)
    largeSampleMatrix, stackedSmall, stackedSmallWhite, labelMatrix, labelStrings = temp

    # Represent (pooled is one example per COLUMN)
    hidden,pooled = tica.getRepresentation(stackedSmallWhite)

    # Create the input data for the next layer.

    # 1. Subsample (via bed of nails, no averaging) each of the four
    # overlapping windows by a factor of two in each direction,
    # creating the same number of total dimensions (exact for even
    # dimensions, approximate for odd).  Example: pooled size is
    # 15x15, downsample to 8x8, stack with four neighbors for a total
    # of 8x8x4 = 256 (instead of 225) dimensions.
    if pooled.shape[0] != hiddenSize**2:
        raise Exception('Expected pooled layer to be %dx%d, but it is length %d'
                        % (hiddenSize, hiddenSize, pooled.shape[0]))
    downsampleSize = (hiddenSize+1)/2    # round up
    downsampleSizeSq = downsampleSize**2
    downsampled = zeros((downsampleSize**2, Nsamples*4))   # store in columns for use with neighbor matrix
    
    gaussNeighbors = neighborMatrix((downsampleSize,downsampleSize), STACKED_NEIGHBORHOOD_SIZE, gaussian=True)
    for smallIdx in xrange(4*Nsamples):
        poolEmbedded = reshape(pooled[:,smallIdx], (hiddenSize, hiddenSize))     # 15x15
        downsampled[:,smallIdx] = poolEmbedded[::2,::2].flatten()                # 8x8 -> 64 column

    # 2. LCN
    vv = downsampled - dot(gaussNeighbors, downsampled)
    sig = sqrt(dot(gaussNeighbors, vv**2))
    cc = .01     # ss = sorted(sig.flatten()); ss[len(ss)/10] = 0.026 in one test. So .01 seems about right.
    yy = vv / maximum(cc, sig)
    
    # 3. Stack together 4x (one example per COLUMN)
    nextLayerInput = zeros((downsampleSizeSq * 4, Nsamples))
    for smallIdx in xrange(4*Nsamples):
        col    = smallIdx / 4
        offset = smallIdx % 4
        nextLayerInput[(offset*downsampleSizeSq):((offset+1)*downsampleSizeSq),col] = yy[:,smallIdx]

    print 'time: %f seconds' % (time.time()-tic)
    
    return nextLayerInput



def main():
    resman.start('junk', diary = False)

    #l1tica  = loadFromPklGz('results/130402_033310_44cc757_master_psearchTica_UP/00022_data/tica.pkl.gz')
    l1tica  = loadFromPklGz('results/130406_184751_3d90386_rapidhacks_upson3_1c_l1/tica.pkl.gz')   # 1c Upson3
    #layer1Whitener = loadFromPklGz('../data/upson_rovio_2/white/train_10_50000_1c.whitener.pkl.gz')
    layer1Whitener = loadFromPklGz('../data/upson_rovio_3/white/train_10_50000_1c.whitener.pkl.gz')

    layerSizePlan = [10, 15, 23, 35, 53, 80, 120, 180]

    stackedTica = StackedTICA(l1tica, layer1Whitener, '../data/upson_rovio_3/imgfiles/',
                              layerSizePlan = layerSizePlan,
                              isColor = False,
                              saveDir = resman.rundir)

    if False:
        print 'JUST DEBUG...'

        pdb.set_trace()
        
        tica = stackedTica.ticas[0]
        pdb.set_trace()
        return
    #data,labels,strings = loadUpsonData3('../data/upson_rovio_3/train_10_50000_1c.pkl.gz')
    #data = loadFromPklGz('../data/upson_rovio_3/white/train_10_50000_1c.white.pkl.gz')
    #stackedTica.plotResults(resman.rundir, stackedTica.ticas[0], data, (10,10), (15,15))
    
    #pdb.set_trace()

    params = {}
    params['hiddenISize'] = 15
    params['hiddenJSize'] = params['hiddenISize']
    params['neighborhoodSize'] = 1.0
    params['lambd'] = .026
    params['randSeed'] = 0
    #params['dataWidth'] = 10
    #params['nColors'] = 1
    #params['isColor'] = (params['nColors'] == 3)
    #params['imgShape'] = ((params['dataWidth'], params['dataWidth'], 3)
    #                      if params['isColor'] else
    #                      (params['dataWidth'], params['dataWidth']))
    params['maxFuncCalls'] = 3
    #params['whiten'] = True    # Just false for Space Invaders dataset...
    params['dataCrop'] = None       # Set to None to not crop data...
    #params['dataCrop'] = 10000       # Set to None to not crop data...

    stackedTica.learnNextLayer(params)
    #print 'HACK FOR DEBUG'
    saveToFile(os.path.join(resman.rundir, 'stackedTica.pkl.gz'), stackedTica)    # save learned model
    
    resman.stop()



if __name__ == '__main__':
    main()
