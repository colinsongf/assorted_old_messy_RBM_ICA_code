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
from GitResultsManager import resman, fmtSeconds

from rica import RICA
from util.plotting import tile_raster_images
from util.dataLoaders import loadFromPklGz, saveToFile



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/atari/mspacman_train_15_50000_3c.pkl.gz')
    data = data.T   # Make into one example per column
    
    random.seed(0)
    rica = RICA(imgShape = (15, 15, 3),
                nFeatures = 400,
                lambd = .05,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    rica.run(data, maxFun = 300, whiten = True)

    resman.stop()
