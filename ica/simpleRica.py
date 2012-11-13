#! /usr/bin/env ipython

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys, time
from numpy import *

from rica import RICA
from util.ResultsManager import resman
from util.dataLoaders import loadFromPklGz, saveToFile



if __name__ == '__main__':
    resman.start('junk', diary = False)

    Nw = 4
    dataXX, dataYY = loadFromPklGz('../data/simpleShapes/train_%02d_50.pkl.gz' % Nw)
    #dataXX, dataYY = loadFromPklGz('../data/simpleShapes/train_%02d_50000.pkl.gz' % Nw)
    data = dataXX.T   # Make into one example per column

    random.seed(0)
    rica = RICA(imgShape = (Nw, Nw),
                nFeatures = 400,
                lambd = .05,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    rica.run(data, maxFun = 300)
    #rica.run(data, maxFun = 300, whiten = True)

    resman.stop()
