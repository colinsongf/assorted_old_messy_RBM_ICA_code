#! /usr/bin/env python

import numpy, time, gzip, PIL.Image, os, pdb
import pickle, sys
from numpy import *

from ResultsManager import resman
from rbm import RBM, test_rbm




def loadPickledData(trainFile, testFile):
    ''' Loads the dataset and returns in the expected train,valid,test format.'''

    # Load the dataset
    ff = gzip.open(trainFile,'rb')
    train_set = pickle.load(ff)
    ff.close()
    ff = gzip.open(testFile,'rb')
    test_set = pickle.load(ff)
    ff.close()

    # no validation set
    return train_set, [array([]), None], test_set



if __name__ == '__main__':
    resman.start('junk', diary = True)

    circles = False
    if len(sys.argv) > 1 and sys.argv[1] == '--circles':
        circles = True
        del sys.argv[1]

    print 'Using dataset:', 'circles' if circles else 'squares'

    img_dim = 10    # 2, 4, 10, 15, 28
    if circles:
        datasets = loadPickledData('../data/circles/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/circles/test_%d_50000.pkl.gz' % img_dim)
    else:
        datasets = loadPickledData('../data/squares/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/squares/test_%d_50000.pkl.gz' % img_dim)
    print 'done loading.'
    test_rbm(datasets = datasets,
             training_epochs = 5,
             img_dim = img_dim,
             n_hidden = 100, 
             learning_rate = .1, 
             output_dir = resman.rundir,
             quickHack = False)
    resman.stop()
