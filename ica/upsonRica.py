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

from rica import RICA
from GitResultsManager import resman, fmtSeconds
from util.plotting import tile_raster_images
from util.dataLoaders import loadFromPklGz, saveToFile



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/upson_rovio_2/train_10_50000_1c.pkl.gz')
    data = data.T   # Make into one example per column
    #data = data[:,:5000]      # HACK!!!!!!!!!
    
    nFeatures = 100
    lambd = .05
    neighborhoodSize = 1.5
    print '\nChosen TICA parameters'
    
    for key in ['nFeatures', 'lambd']:
        print '  %20s: %s' % (key, locals()[key])
    
    random.seed(0)
    rica = RICA(imgShape = (10, 10),
                nFeatures = nFeatures,
                lambd = lambd,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    rica.run(data, maxFun = 300, whiten = True, plotEvery = 50)


    resman.stop()
