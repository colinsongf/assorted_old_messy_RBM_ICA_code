#! /usr/bin/env python

import os, pdb, gzip, sys
import argparse
from numpy import *

from utils import loadFromFile
from squaresRbm import loadPickledData
from ResultsManager import resman
from SVMLightWrapper import SVMLightWrapper
from generativeLearn import makeSizedDataset



def main(traindata0, traindata1, testdata0, testdata1, rundir):
    random.seed(0)

    xx0train, yy0train = loadPickledData(traindata0)
    xx1train, yy1train = loadPickledData(traindata1)
    xx0test,  yy0test  = loadPickledData(testdata0)
    xx1test,  yy1test  = loadPickledData(testdata1)

    testX,  testY  = makeSizedDataset(1000,
                                      ((xx0test,None),(None,None),(None,None)),
                                      ((xx1test,None),(None,None),(None,None)),
                                      appendClass = False)
    
    sizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000]
    #sizes = [10, 20, 40, 100]
    errorRates = []

    for size in sizes:
        trainX, trainY = makeSizedDataset(size,
                                          ((xx0train,None),(None,None),(None,None)),
                                          ((xx1train,None),(None,None),(None,None)),
                                          appendClass = False)
        svm = SVMLightWrapper(kernelType = 0, C = 1, z = 'c')

        #trainX = vstack((xx0train, xx1train))
        #trainY = array(vstack((zeros((xx0train.shape[0], 1)), ones((xx1train.shape[0], 1)))), dtype = bool)
        #testX = vstack((xx0test, xx1test))
        #testY = array(vstack((zeros((xx0test.shape[0], 1)), ones((xx1test.shape[0], 1)))), dtype = bool)

        print 'training...'
        svm.train(trainX, trainY)
        print 'done'
        print 'predicting...'
        predY = svm.predict(testX)
        print 'done'

        print testY.flatten()
        print predY > 0


        errorRate = float(sum(testY.flatten() != (predY > 0))) / len(predY)
        errorRates.append(errorRate)
        print 'Size', size, 'Error rate:', errorRate
        print 'so far:', errorRates

    print 'Sizes:', sizes
    print 'Error rates:', errorRates



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify data using an SVM')
    parser.add_argument('traindata0', metavar = 'filename', type = str,
                        help='Class 0 train set')
    parser.add_argument('traindata1', metavar = 'filename', type = str,
                        help='Class 1 train set')
    parser.add_argument('testdata0', metavar = 'filename', type = str,
                        help='Class 0 test set')
    parser.add_argument('testdata1', metavar = 'filename', type = str,
                        help='Class 1 test set')
    parser.add_argument('--name', metavar = 'name', type = str,
                        default = 'junk',
                        help='Name of run for ResultsManager. Default: junk')
    args = parser.parse_args()

    resman.start(args.name, diary = False)
    
    main(args.traindata0, args.traindata1, args.testdata0, args.testdata1,
         rundir = resman.rundir)

    resman.stop()

