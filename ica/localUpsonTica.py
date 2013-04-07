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
#from IPython.parallel import Client

from tica import TICA
from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, loadUpsonData, loadUpsonData3, loadRandomData, saveToFile
from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
from util.misc import pt, pc
from paramSearchTica import runTest



def main():
    resman.start('junk', diary = True)

    params = {}
    randomParams = False
    if randomParams:
        params['hiddenISize'] = random.choice((2, 4, 6, 8, 10, 15, 20))
        params['hiddenJSize'] = params['hiddenISize']
        params['neighborhoodSize'] = random.choice((.1, .3, .5, .7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0))
        lambd = exp(random.uniform(log(.0001), log(10)))   # Uniform in log space
        params['lambd'] = round(lambd, 1-int(floor(log10(lambd))))  # Just keep two significant figures
        params['randSeed'] = int(random.uniform(0,9999))
        #params['dataWidth'] = random.choice((2, 4))   # just quick
        #params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20, 25, 28))
        params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20))  # 25 and 28 are incomplete
        params['nColors'] = random.choice((1, 3))
    else:
        params['hiddenISize'] = 15
        params['hiddenJSize'] = params['hiddenISize']
        params['neighborhoodSize'] = 1.0
        params['lambd'] = .026
        params['randSeed'] = 22
        #params['dataWidth'] = random.choice((2, 4))   # just quick
        #params['dataWidth'] = random.choice((2, 3, 4, 6, 10, 15, 20, 25, 28))
        params['dataWidth'] = 10
        params['nColors'] = 3
    params['isColor'] = (params['nColors'] == 3)
    params['imgShape'] = ((params['dataWidth'], params['dataWidth'], 3)
                          if params['isColor'] else
                          (params['dataWidth'], params['dataWidth']))
    params['maxFuncCalls'] = 300
    params['whiten'] = True    # Just false for Space Invaders dataset...
    params['dataCrop'] = None       # Set to None to not crop data...

    paramsRand = params.copy()
    paramsRand['dataLoader'] = 'loadRandomData'
    paramsRand['dataPath'] = ('../data/random/randomu01_train_%02d_50000_%dc.pkl.gz'
                              % (paramsRand['dataWidth'], paramsRand['nColors']))

    paramsData = params.copy()
    #paramsData['dataLoader'] = 'loadAtariData'
    #paramsData['dataPath'] = ('../data/atari/space_invaders_train_%02d_50000_%dc.pkl.gz'
    #                          % (paramsData['dataWidth'], paramsData['nColors']))
    paramsData['dataLoader'] = 'loadUpsonData3'
    paramsData['dataPath'] = ('../data/upson_rovio_3/train_%02d_50000_%dc.pkl.gz'
                              % (paramsData['dataWidth'], paramsData['nColors']))
        

    doRand = False
    if doRand:
        #resultsRand = reliablyRunTest((0, resman.rundir, '00000_rand', paramsRand, os.getcwd(), os.getenv('DISPLAY','')))
        randResultsDir = os.makedirs(resman.rundir, 'rand')
        runTest(randResultsDir, paramsRand)
    #resultsData = reliablyRunTest((0, resman.rundir, '00000_data', paramsData, os.getcwd(), os.getenv('DISPLAY','')))

    runTest(resman.rundir, paramsData)
    
    resman.stop()
    

if __name__ == '__main__':
    main()
