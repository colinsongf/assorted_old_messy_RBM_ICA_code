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

from tica import TICA
from visualize import plotImageData, plotCov, printDataStats, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, loadRandomData, saveToFile
from util.dataPrep import PCAWhiteningDataNormalizer
from util.misc import pt, pc



#from IPython.parallel import require
#@require(util.misc)
def runTest(args):
    testId, rundir, childdir, params, cwd = args
    
    import time, os
    os.chdir(cwd)
    from numpy import *
    from GitResultsManager import GitResultsManager
    #raise Exception('cwd is %s' % os.getcwd())
    from tica import TICA
    from util.misc import MakePc, Counter
    from visualize import plotImageData, plotCov, printDataStats, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
    from util.dataPrep import PCAWhiteningDataNormalizer
    from util.dataLoaders import loadAtariData, loadRandomData, saveToFile
    #counter = Counter()
    #pc = lambda st : makePc(st, counter = counter)
    pc = MakePc(Counter())
    
    childResman = GitResultsManager()
    childResman.start(childOfRunDir = rundir, description = childdir, diary = False)
    
    saveDir = childResman.rundir

    dataCrop = None
    #########################
    # Parameters
    #########################
    hiddenISize        = params['hiddenISize']
    hiddenJSize        = params['hiddenJSize']
    neighborhoodParams = ('gaussian', params['neighborhoodSize'], 0, 0)
    lambd              = params['lambd']
    epsilon            = 1e-5
    maxFuncCalls       = params['maxFuncCalls']
    randSeed           = params['randSeed']
    dataCrop           = 1000

    dataLoader         = locals().get(params['dataLoader'])  # Convert string to actual function
    dataPath           = params['dataPath']
    imgShape           = params['imgShape']

    hiddenLayerShape = (hiddenISize, hiddenJSize)

    #########################
    # Data
    #########################

    # Load data
    #data = loadAtariData('../data/atari/mspacman_train_15_50000_3c.pkl.gz'); imgShape = (15,15,3)
    #data = loadAtariData('../data/atari/space_invaders_train_15_50000_3c.pkl.gz'); imgShape = (15,15,3)
    data = dataLoader(dataPath)
    if dataCrop:
        print '\nWARNING: Cropping data from %d examples to only %d for debug\n' % (data.shape[1], dataCrop)
        data = data[:,:dataCrop]
    nInputs = data.shape[0]
    isColor = len(imgShape) > 2

    print '\nParameters:'
    for key in ['nInputs', 'hiddenISize', 'hiddenJSize', 'neighborhoodParams', 'lambd', 'epsilon', 'maxFuncCalls', 'randSeed', 'dataCrop', 'dataLoader', 'dataPath', 'imgShape']:
        print '  %20s: %s' % (key, locals()[key])
    print

    skipVis = True
    if not skipVis:
        # Visualize before prep
        plotImageData(data, imgShape, saveDir, pc('data_raw'))
        plotCov(data, saveDir, pc('data_raw'))
    printDataStats(data)
    
    # Whiten with PCA
    whiteningStage = PCAWhiteningDataNormalizer(data, unitNorm = True, saveDir = saveDir)
    dataWhite, junk = whiteningStage.raw2normalized(data)
    dataOrig        = whiteningStage.normalized2raw(dataWhite)

    if not skipVis:
        # Visualize after prep
        plotImageData(dataWhite, imgShape, saveDir, pc('data_white'))
        plotCov(dataWhite, saveDir, pc('data_white'))
    printDataStats(dataWhite)


    #########################
    # Model
    #########################

    random.seed(randSeed)
    tica = TICA(nInputs            = prod(imgShape),
                hiddenLayerShape   = hiddenLayerShape,
                neighborhoodParams = neighborhoodParams,
                lambd              = lambd,
                epsilon            = epsilon,
                saveDir            = saveDir)

    beginTotalCost, beginPoolingCost, beginReconstructionCost, grad = tica.cost(tica.WW, data)

    tic = time.time()
    tica.learn(dataWhite, maxFun = maxFuncCalls)
    execTime = time.time() - tic
    saveToFile(os.path.join(saveDir, 'tica.pkl.gz'), tica)    # save learned model

    plotImageRicaWW(tica.WW, imgShape, saveDir, tileShape = hiddenLayerShape, prefix = pc('WW_iterFinal'))
    plotRicaActivations(tica.WW, dataWhite, saveDir, prefix = pc('activations_iterFinal'))
    plotRicaReconstructions(tica, dataWhite, imgShape, saveDir, unwhitener = whiteningStage.normalized2raw,
                            tileShape = hiddenLayerShape, prefix = pc('recon_iterFinal'),
                            number = 20)

    endTotalCost, endPoolingCost, endReconstructionCost, grad = tica.cost(tica.WW, data)

    print 'beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime ='
    print [beginTotalCost, beginPoolingCost, beginReconstructionCost, endTotalCost, endPoolingCost, endReconstructionCost, execTime]
    results = {'beginTotalCost': beginTotalCost,
               'beginPoolingCost': beginPoolingCost,
               'beginReconstructionCost': beginReconstructionCost,
               'endTotalCost': endTotalCost,
               'endPoolingCost': endPoolingCost,
               'endReconstructionCost': endReconstructionCost,
               'execTime': execTime}
    childResman.stop()

    return testId, params, results



def main():
    resman.start('junk', diary = False)

    client = Client()
    print 'IPython worker ids:', client.ids
    balview = client.load_balanced_view()

    resultsFilename = os.path.join(resman.rundir, 'allResults.pkl.gz')
    
    NN = 2
    allResults = [[None,None] for ii in range(NN)]

    experiments = []
    cwd = os.getcwd()
    useIpython = True
    for ii in range(NN):
        params = {}
        random.seed(ii)
        params['hiddenISize'] = random.choice((2, 4, 6, 8, 10, 15, 20))
        params['hiddenJSize'] = params['hiddenISize']
        params['neighborhoodSize'] = random.choice((.1, .3, .5, .7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0))
        lambd = exp(random.uniform(log(.0001), log(10)))   # Uniform in log space
        params['lambd'] = round(lambd, 1-int(floor(log10(lambd))))  # Just keep two significant figures
        params['randSeed'] = ii
        params['maxFuncCalls'] = 2
        #params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20, 25, 28))
        params['dataWidth'] = random.choice((2, 4, 10))   # just quick
        params['imgShape'] = (params['dataWidth'], params['dataWidth'], 3)   # use color

        paramsRand = params.copy()
        paramsRand['dataLoader'] = 'loadRandomData'
        paramsRand['dataPath'] = '../data/random/randomu01_train_%02d_50000_3c.pkl.gz' % paramsRand['dataWidth']

        paramsData = params.copy()
        paramsData['dataLoader'] = 'loadAtariData'
        paramsData['dataPath'] = '../data/atari/space_invaders_train_%02d_50000_3c.pkl.gz' % paramsData['dataWidth']

        if not useIpython:
            resultsRand = runTest(resman.rundir, '%05d_rand' % ii, paramsRand)

            allResults[ii][0] = {'params': paramsRand, 'results': resultsRand}
            tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
            saveToFile(tmpFilename, allResults)
            os.rename(tmpFilename, resultsFilename)

            resultsData = runTest(resman.rundir, '%05d_data' % ii, paramsData)

            allResults[ii][1] = {'params': paramsData, 'results': resultsData}
            tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
            saveToFile(tmpFilename, allResults)
            os.rename(tmpFilename, resultsFilename)
        else:
            experiments.append(((ii, 0), resman.rundir, '%05d_rand' % ii, paramsRand, cwd))
            experiments.append(((ii, 1), resman.rundir, '%05d_data' % ii, paramsData, cwd))


    # Start all jobs
    jobMap = balview.map_async(runTest, experiments, ordered = False)
    #jobMap = balview.map_async(runTest, range(10), ordered = False)
    for ii, returnValues in enumerate(jobMap):
        testId, params, results = returnValues
        print ii, 'Job', testId, 'finished.'
        allResults[testId[0]][testId[1]] = {'params': params, 'results': results}
        tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
        saveToFile(tmpFilename, allResults)
        os.rename(tmpFilename, resultsFilename)

    print 'Finished all jobs.'
    resman.stop()
    

if __name__ == '__main__':
    main()
