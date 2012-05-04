#! /usr/bin/env python

import numpy, time, gzip, PIL.Image, os, pdb
import pickle, sys
from numpy import *

from ResultsManager import resman
from squaresRbm import loadPickledData
from rbm import RBM, test_rbm



if __name__ == '__main__':
    resman.start('junk', diary = True)

    spheres = False
    if len(sys.argv) > 1 and sys.argv[1] == '--spheres':
        spheres = True
        del sys.argv[1]

    print 'Using dataset:', 'spheres' if spheres else 'cubes'

    img_dim = 10    # 2, 4, 10, 15, 28
    if spheres:
        datasets = loadPickledData('../data/spheres/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/spheres/test_%d_50000.pkl.gz' % img_dim)
    else:
        datasets = loadPickledData('../data/cubes/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/cubes/test_%d_50000.pkl.gz' % img_dim)
    print 'done loading.'
    rbm, meanCosts = test_rbm(datasets = datasets,
                              training_epochs = 45,
                              img_dim = img_dim,
                              n_hidden = 200, 
                              learning_rate = .1, 
                              output_dir = resman.rundir,
                              quickHack = false,
                              imgPlotFunction = lambda xx: xx[:,0:img_dim*img_dim],  # HACK: plot first slice
                              )
    resman.stop()
