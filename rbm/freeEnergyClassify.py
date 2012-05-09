#! /usr/bin/env python

import os, pdb, gzip, sys
import argparse
from numpy import *
from tvtk.api import tvtk

from utils import loadFromFile
from squaresRbm import loadPickledData
from ResultsManager import resman



def main(rbmFilenames, data0Filename, data1Filename, rundir):
    random.seed(0)

    xx0, yy0 = loadPickledData(data0Filename)
    xx1, yy1 = loadPickledData(data1Filename)

    testX = vstack((xx0, xx1))
    testY = array(vstack((zeros((xx0.shape[0], 1)), ones((xx1.shape[0], 1)))), dtype = bool)
    testWhole = hstack((testX, testY))

    # shuffle order of dataset
    random.shuffle(testWhole)
        
    testX = testWhole[:,:-1]
    testY = array(testWhole[:,-1], dtype='uint8')

    errorRates = []
    for rbmFilename in rbmFilenames:
        print rbmFilename
        
        rbm = loadFromFile(rbmFilename)
        pred  = -1 * ones(testY.shape, dtype='uint8')

        batchSize = 100
        if len(pred) % batchSize != 0:
            raise Exception('must be a multiple of batchSize (%d)!' % batchSize)

        for ii in xrange(testY.shape[0] / batchSize):
        #for bb in xrange(2000):
            print ii * batchSize
            testPt0 = hstack((testX[ii*batchSize:(ii+1)*batchSize,:], zeros((batchSize,1))))
            testPt1 = hstack((testX[ii*batchSize:(ii+1)*batchSize,:], ones((batchSize,1))))
            fe0 = rbm.free_energy(testPt0)
            fe1 = rbm.free_energy(testPt1)
            #print 'y', testY[ii], 'fe0', fe0, 'fe1', fe1, 'diff', fe0 - fe1, '0' if fe0 < fe1 else '1'

            pred[ii*batchSize:(ii+1)*batchSize] = 1 * (fe1 > fe0)
            #if ii % 1000 == 0:
            #    print ii

            #testPt0 = hstack((testX[ii,:], 0))
            #testPt1 = hstack((testX[ii,:], 1))
            #fe0 = rbm.free_energy(testPt0)
            #fe1 = rbm.free_energy(testPt1)
            #print 'y', testY[ii], 'fe0', fe0, 'fe1', fe1, 'diff', fe0 - fe1, '0' if fe0 < fe1 else '1'
            ##pred[ii] = 0 if fe0 < fe1 else 1
            #pred[ii] = 0 if fe0 > fe1 else 1

        ##pred[ii] = 0 if fe0 < fe1 else 1
        #
        ##if ii % 1000 == 0:
        ##    print ii

        errorRate = float(sum(pred != testY)) / len(testY)
        errorRates.append(errorRate)

    print 'Error rates:', errorRates

    #pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify data using the free energy of the RBM. This program assumes that the RBM was trained with a 0 or 1 appended to the data vector')
    parser.add_argument('data0', metavar = 'filename', type = str,
                        help='Data filename to load for class 0')
    parser.add_argument('data1', metavar = 'filename', type = str,
                        help='Data filename to load for class 1')
    parser.add_argument('rbm', metavar = 'filename', type = str,
                        help='RBM to load from .pkl.gz file', nargs = '+')
    parser.add_argument('--name', metavar = 'name', type = str,
                        default = 'junk',
                        help='Name of run for ResultsManager. Default: junk')
    args = parser.parse_args()

    resman.start(args.name, diary = False)
    
    main(args.rbm, args.data0, args.data1,
         rundir = resman.rundir)

    resman.stop()
    
