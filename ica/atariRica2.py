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

from rica import RICA
from visualize import plotImageData, plotCov, printDataStats, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, saveToFile
from util.dataPrep import PCAWhiteningDataNormalizer
from util.misc import pt, pc



'''Refactored version'''

if __name__ == '__main__':
    resman.start('junk', diary = False)
    saveDir = resman.rundir

    dataCrop = None
    #########################
    # Parameters
    #########################
    nFeatures    = 400
    lambd        = .05
    epsilon      = 1e-5
    maxFuncCalls = 3
    randSeed     = 0
    #dataCrop    = 1000
    
    #########################
    # Data
    #########################

    # Load data
    data = loadAtariData('../data/atari/mspacman_train_15_50000_3c.pkl.gz'); imgShape = (15,15,3)
    if dataCrop:
        print 'Warning: Cropping data from %d examples to only %d for debug' % (data.shape[1], dataCrop)
        data = data[:,:dataCrop]
    nInputs = data.shape[0]
    isColor = len(imgShape) > 2

    print '\nParameters:'
    for key in ['nInputs', 'nFeatures', 'lambd', 'epsilon', 'maxFuncCalls', 'randSeed', 'dataCrop']:
        print '  %20s: %s' % (key, locals()[key])
    print

    # Visualize before prep
    plotImageData(data, imgShape, saveDir, pc('data_raw'))
    plotCov(data, saveDir, pc('data_raw'))
    printDataStats(data)
    
    # Whiten with PCA
    whiteningStage = PCAWhiteningDataNormalizer(data, unitNorm = True, saveDir = saveDir)
    dataWhite, junk = whiteningStage.raw2normalized(data)
    dataOrig        = whiteningStage.normalized2raw(dataWhite)

    # Visualize after prep
    plotImageData(dataWhite, imgShape, saveDir, pc('data_white'))
    plotCov(dataWhite, saveDir, pc('data_white'))
    printDataStats(dataWhite)


    #########################
    # Model
    #########################

    random.seed(randSeed)
    rica = RICA(nInputs   = prod(imgShape),
                nOutputs  = nFeatures,
                lambd     = lambd,
                epsilon   = epsilon,
                saveDir   = saveDir)

    plotImageRicaWW(rica.WW, imgShape, saveDir, prefix = pc('WW_iter0'))
    plotRicaActivations(rica.WW, dataWhite, saveDir, prefix = pc('activations_iter0'))
    plotRicaReconstructions(rica, dataWhite, imgShape, saveDir, unwhitener = whiteningStage.normalized2raw, prefix = pc('recon_iter0'))
    
    rica.learn(dataWhite, maxFun = maxFuncCalls)
    saveToFile(os.path.join(saveDir, 'rica.pkl.gz'), rica)    # save learned model

    plotImageRicaWW(rica.WW, imgShape, saveDir, prefix = pc('WW_iterFinal'))
    plotRicaActivations(rica.WW, dataWhite, saveDir, prefix = pc('activations_iterFinal'))
    plotRicaReconstructions(rica, dataWhite, imgShape, saveDir, unwhitener = whiteningStage.normalized2raw, prefix = pc('recon_iterFinal'))
    
    resman.stop()
