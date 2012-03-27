#! /usr/bin/env python

# [JBY] copied from RBM.py
# [JBY] From http://deeplearning.net/tutorial/rbm.html



import numpy, time, gzip, PIL.Image, os, pdb
#import cPickle as pickle
import pickle
from numpy import *

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import resman
from rbm import RBM, test_rbm




def loadUpsonData(trainFile, testFile):
    ''' Loads the dataset and returns in the expected train,valid,test format.'''

    # Load the dataset
    ff = gzip.open(trainFile,'rb')
    train_set = pickle.load(ff)
    ff.close()
    ff = gzip.open(testFile,'rb')
    test_set = pickle.load(ff)
    ff.close()

    # no validation set, no y (purely unsupervised)
    return [train_set, None], [array([]), None], [test_set, None]



if __name__ == '__main__':
    resman.start('junk', diary = False)
    datasets = loadUpsonData('../data/upson_rovio_1/train_10_50000.pkl.gz',
                             '../data/upson_rovio_1/test_10_50000.pkl.gz')
    print 'done loading.'
    test_rbm(datasets = datasets,
             training_epochs = 45,
             img_dim = 10,
             n_hidden = 500,
             learning_rate = .002,
             output_folder = resman.rundir,
             quickHack = False)
    resman.stop()
