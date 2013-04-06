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
from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, loadUpsonData, loadUpsonData3, loadRandomData, saveToFile
from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
from util.misc import pt, pc



#from IPython.parallel import require
#@require(util.misc)
def reliablyRunTest(args):
    import os
    import time
    import traceback
    from GitResultsManager import GitResultsManager, hostname
    from numpy import random

    testId, rundir, childdir, params, cwd, display = args
    
    os.chdir(cwd)
    os.putenv('DISPLAY', display)   # needed on Xanthus

    from paramSearchTica import runTest

    childResman = GitResultsManager()
    childResman.start(childOfRunDir = rundir, description = childdir, diary = True)  # Turn off for now :(
    saveDir = childResman.rundir
    
    tries = 0
    myHostname = hostname()
    results = {}
    results['failures'] = []
    results['host'] = myHostname
    success = False
    maxTries = 10
    while not success and tries < maxTries:
        tic = time.time()
        tries += 1
        print 'reliablyRunTest: Attempt %d' % tries
        try:
            runResults = runTest(saveDir, params)
            success = True
        except Exception, err:
            print ('reliablyRunTest: FAILURE: runTest on host %s failed on try %d after %f seconds. Error was:'
                   % (myHostname, tries, time.time()-tic))
            tb = traceback.format_exc()
            print tb
            results['failures'].append(tb)
            minSleep = 1.5**tries
            maxSleep = 1.5**(tries+1)
            sleepTime = random.uniform(minSleep, maxSleep)
            print 'Sleeping for %.2f seconds.' % sleepTime
            time.sleep(sleepTime)
    results['success'] = success
    if success:
        results.update(runResults)
    else:
        print 'reliablyRunTest: Giving up after %d tries' % tries

    childResman.stop()

    return testId, params, results


def runTest(saveDir, params):

    import time, os, sys
    from numpy import *
    from GitResultsManager import GitResultsManager
    #raise Exception('path is %s' % sys.path)
    #raise Exception('version is %s' % sys.version_info)
    #raise Exception('cwd is %s' % os.getcwd())
    from tica import TICA
    from util.misc import MakePc, Counter
    from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
    from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
    from util.dataLoaders import loadAtariData, loadUpsonData, loadUpsonData3, loadRandomData, saveToFile
    #counter = Counter()
    #pc = lambda st : makePc(st, counter = counter)
    pc = MakePc(Counter())
    
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
    whiten             = params['whiten']
    dataCrop           = params['dataCrop']

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
    loaded = dataLoader(dataPath)
    if type(loaded) is tuple:
        data, labels, labelStrings = loaded
        print 'Data has labels:', labelStrings
    else:
        data = loaded
        labels, labelStrings = None, None
        print 'Data does not have labels.'
    if dataCrop:
        print '\nWARNING: Cropping data from %d examples to only %d for debug\n' % (data.shape[1], dataCrop)
        data = data[:,:dataCrop]
    nInputs = data.shape[0]
    isColor = len(imgShape) > 2

    print '\nParameters:'
    for key in ['nInputs', 'hiddenISize', 'hiddenJSize', 'neighborhoodParams', 'lambd', 'epsilon', 'maxFuncCalls', 'randSeed', 'dataCrop', 'dataLoader', 'dataPath', 'imgShape', 'whiten']:
        print '  %20s: %s' % (key, locals()[key])
    print

    skipVis = True
    if whiten:
        if not skipVis:
            # Visualize before prep
            plotImageData(data, imgShape, saveDir, pc('data_raw'))
            plotCov(data, saveDir, pc('data_raw'))
        printDataStats(data)

        # Whiten with PCA
        whiteningStage = PCAWhiteningDataNormalizer(data, saveDir = saveDir)
        dataWhite, junk = whiteningStage.raw2normalized(data, unitNorm = True)
        #dataOrig        = whiteningStage.normalized2raw(dataWhite)
        dataOrig = data
        data = dataWhite

    if not skipVis:
        # Visualize after prep
        plotImageData(data, imgShape, saveDir, pc('data_white'))
        plotCov(data, saveDir, pc('data_white'))
    printDataStats(data)


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
    tica.learn(data, maxFun = maxFuncCalls)
    execTime = time.time() - tic
    saveToFile(os.path.join(saveDir, 'tica.pkl.gz'), tica)    # save learned model

    plotImageRicaWW(tica.WW, imgShape, saveDir, tileShape = hiddenLayerShape, prefix = pc('WW_iterFinal'))
    plotRicaActivations(tica.WW, data, saveDir, prefix = pc('activations_iterFinal'))
    unwhitener = whiteningStage.normalized2raw if whiten else None
    plotRicaReconstructions(tica, data, imgShape, saveDir,
                            unwhitener = unwhitener,
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

    # Save locally just in case of exception in main program
    myResults = {'params': params, 'results': results}
    saveToFile(os.path.join(saveDir, 'myresults.pkl.gz'), myResults)
    
    return results



def main():
    resman.start('junk', diary = False)

    useIpython = True
    if useIpython:
        client = Client(profile='ssh')
        #client = Client()
        print 'IPython worker ids:', client.ids
        balview = client.load_balanced_view()

    resultsFilename = os.path.join(resman.rundir, 'allResults.pkl.gz')
    
    NN = 1000
    allResults = [[None,None] for ii in range(NN)]

    experiments = []
    cwd = os.getcwd()
    disp = os.environ['DISPLAY']
    for ii in range(NN):
        params = {}
        random.seed(ii)
        params['hiddenISize'] = random.choice((2, 4, 6, 8, 10, 15, 20))
        params['hiddenJSize'] = params['hiddenISize']
        params['neighborhoodSize'] = random.choice((.1, .3, .5, .7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0))
        lambd = exp(random.uniform(log(.0001), log(10)))   # Uniform in log space
        params['lambd'] = round(lambd, 1-int(floor(log10(lambd))))  # Just keep two significant figures
        params['randSeed'] = ii
        params['maxFuncCalls'] = 300
        #params['dataWidth'] = random.choice((2, 4))   # just quick
        #params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20, 25, 28))
        params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20))  # 25 and 28 are incomplete
        params['nColors'] = random.choice((1, 3))
        params['isColor'] = (params['nColors'] == 3)
        params['imgShape'] = ((params['dataWidth'], params['dataWidth'], 3)
                              if params['isColor'] else
                              (params['dataWidth'], params['dataWidth']))
        params['whiten'] = False    # Just false for Space Invaders dataset...
        params['dataCrop'] = None       # Set to None to not crop data...
        
        paramsRand = params.copy()
        paramsRand['dataLoader'] = 'loadRandomData'
        paramsRand['dataPath'] = ('../data/random/randomu01_train_%02d_50000_%dc.pkl.gz'
                                  % (paramsRand['dataWidth'], paramsRand['nColors']))

        paramsData = params.copy()
        paramsData['dataLoader'] = 'loadAtariData'
        paramsData['dataPath'] = ('../data/atari/space_invaders_train_%02d_50000_%dc.pkl.gz'
                                  % (paramsData['dataWidth'], paramsData['nColors']))
        #paramsData['dataLoader'] = 'loadUpsonData'
        #paramsData['dataPath'] = ('../data/upson_rovio_2/train_%02d_50000_%dc.pkl.gz'
        #                          % (paramsData['dataWidth'], paramsData['nColors']))

        if not useIpython:
            resultsRand = reliablyRunTest(resman.rundir, '%05d_rand' % ii, paramsRand)

            allResults[ii][0] = {'params': paramsRand, 'results': resultsRand}
            tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
            saveToFile(tmpFilename, allResults)
            os.rename(tmpFilename, resultsFilename)

            resultsData = reliablyRunTest(resman.rundir, '%05d_data' % ii, paramsData)

            allResults[ii][1] = {'params': paramsData, 'results': resultsData}
            tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
            saveToFile(tmpFilename, allResults)
            os.rename(tmpFilename, resultsFilename)
        else:
            experiments.append(((ii, 0), resman.rundir, '%05d_rand' % ii, paramsRand, cwd, disp))
            experiments.append(((ii, 1), resman.rundir, '%05d_data' % ii, paramsData, cwd, disp))


    # Start all jobs
    jobMap = balview.map_async(reliablyRunTest, experiments, ordered = False)
    #jobMap = balview.map_async(reliablyRunTest, range(10), ordered = False)
    for ii, returnValues in enumerate(jobMap):
        testId, params, results = returnValues
        print ii, 'Job', testId, 'finished.'
        allResults[testId[0]][testId[1]] = {'params': params, 'results': results}
        tmpFilename = os.path.join(resman.rundir, '.tmp.%f.pkl.gz' % time.time())
        saveToFile(tmpFilename, allResults)
        os.rename(tmpFilename, resultsFilename)
        #pdb.set_trace()

    print 'Finished all jobs.'
    resman.stop()
    

if __name__ == '__main__':
    main()
