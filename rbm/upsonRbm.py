#! /usr/bin/env python



import numpy, time, gzip, PIL.Image, os, pdb
#import cPickle as pickle
import pickle
from numpy import *

from ResultsManager import resman
from rbm import RBM, test_rbm
from utils import loadUpsonData



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
             output_dir = resman.rundir,
             quickHack = False)
    resman.stop()
