#! /usr/bin/env ipython

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys, time
from numpy import *
from PIL import Image
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

from tica import TICA
from GitResultsManager import resman, fmtSeconds
from util.plotting import tile_raster_images
from util.dataLoaders import loadFromPklGz, saveToFile



if __name__ == '__main__':
    resman.start('junk', diary = False)

    print 'probably want simple instead... finish this file first'
    sys.exit(1)
    data = loadFromPklGz('../data/atari/mspacman_train_15_50000_3c.pkl.gz')
    data = data.T   # Make into one example per column
    #data = data[:,:5000]      # HACK!!!!!!!!!
    
    #hiddenISize = 20
    #hiddenJSize = 20
    #lambd = .05
    #neighborhoodSize = 1
    #print '\nChosen TICA parameters'
    
    hiddenISize = random.randint(4, 25+1)
    hiddenJSize = random.randint(10, 30+1)
    lambd = .1 * 2 ** random.randint(-5, 5+1)
    neighborhoodSize = random.uniform(.1,3)
    print '\nRandomly selected TICA parameters'

    for key in ['hiddenISize', 'hiddenJSize', 'lambd', 'neighborhoodSize']:
        print '  %20s: %s' % (key, locals()[key])
    
    random.seed(0)
    tica = TICA(imgShape = (15, 15, 3),
                hiddenLayerShape = (hiddenISize, hiddenJSize),
                neighborhoodParams = ('gaussian', neighborhoodSize, 0, 0),
                lambd = lambd,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    tica.run(data, plotEvery = 5, maxFun = 300, whiten = True)

    resman.stop()

