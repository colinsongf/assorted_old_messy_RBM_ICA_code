#! /usr/bin/env ipython

'''
Research code

Jason Yosinski
'''

import ipdb as pdb
import os, sys, time
from numpy import *
from PIL import Image
from GitResultsManager import resman
from IPython.parallel import Client

from tica import neighborMatrix, TICA
from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions, plotTopActivations
from util.dataLoaders import loadAtariData, loadUpsonData, loadUpsonData3, loadRandomData, saveToFile, loadFromPklGz
from util.cache import cached
from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
from util.misc import pt, pc
from makeData.makeUpsonRovio3 import randomSampleMatrixWithLabels, trainFilter, testFilter, exploreFilter
from stackedTica import StackedTICA, getBigAndSmallerSamples



def main():
    resman.start('junk', diary = False)

    stica = loadFromPklGz('results/130407_132841_76b6586_rapidhacks_upson3_1c_l2_first/stackedTica_mod.pkl.gz')
    layer1Whitener = loadFromPklGz('../data/upson_rovio_3/white/train_10_50000_1c.whitener.pkl.gz')
    
    layerSizePlan = [10, 15, 23, 35, 53, 80, 120, 180]

    visLayer = 1

    largeSampleMatrix, labelMatrix, labelStrings = cached(randomSampleMatrixWithLabels, trainFilter,
                                                          seed = 0, color = False,
                                                          Nw = layerSizePlan[visLayer], Nsamples = 50000)

    seed = 0
    Nw = layerSizePlan[visLayer-1]             # e.g. 10
    Nwbig = layerSizePlan[visLayer]            # e.g. 15
    Nwshift = Nwbig - Nw                       # e.g. 15 - 10 = 5
    Nsamples = 1000
    temp = getBigAndSmallerSamples(trainFilter, layer1Whitener, seed, False, Nw, Nwshift, Nsamples)
    largeSampleMatrix, stackedSmall, stackedSmallWhite, labelMatrix, labelStrings = temp

    pooled = stica.getRepresentation(largeSampleMatrix)
    
    plotTopActivations(pooled, largeSampleMatrix, (Nwbig,Nwbig), resman.rundir, nActivations = 50, nSamples = 20)

    pl = (pooled.T - pooled.mean(1)).T
    for ii in range(len(labelStrings)):
        print 'finding top for', labelStrings[ii]
        if labelMatrix[ii,:].sum() == 0:
            print '  skipping, no examples'
            continue
        avgActivationForClass = (pl * labelMatrix[ii,:]).mean(1)
        sortIdx = argsort(avgActivationForClass)
        topNeurons = sortIdx[-1:-(50+1):-1]
        plotTopActivations(pooled[topNeurons,:], largeSampleMatrix, (Nwbig,Nwbig), resman.rundir,
                           nActivations = 50, nSamples = 20, prefix = 'topfor_%s' % labelStrings[ii])
        
    resman.stop()



if __name__ == '__main__':
    main()
