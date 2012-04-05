#! /usr/bin/env python



import numpy, time, gzip, PIL.Image, os, pdb
import pickle
from numpy import *

from ResultsManager import resman
from rbm import RBM, test_rbm




def loadSquaresData(trainFile, testFile):
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
    resman.start('junk', diary = False)
    datasets = loadSquaresData('../data/squares/train_10_50000.pkl.gz',
                               '../data/squares/test_10_50000.pkl.gz')
    print 'done loading.'
    test_rbm(datasets = datasets,
             training_epochs = 5,
             img_dim = 10,
             n_hidden = 100, 
             learning_rate = .1, 
             output_dir = resman.rundir,
             quickHack = False)
    resman.stop()
