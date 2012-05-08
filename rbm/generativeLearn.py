#! /usr/bin/env python

import numpy, time, gzip, PIL.Image, os, pdb
import pickle, sys
from numpy import *

from ResultsManager import resman
from squaresRbm import loadPickledData
from rbm import RBM, test_rbm




def makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass):
    random.seed(0)

    # make pseudo dataset with squares vs. spheres labled
    trainX = vstack((cubeDatasets[0][0][0:size,:], sphereDatasets[0][0][0:size,:]))
    trainX = array(trainX, dtype = bool)
    trainY = array(vstack((zeros((size, 1)), ones((size, 1)))), dtype = bool)

    trainWhole = hstack((trainX, trainY))

    # shuffle order of dataset
    random.shuffle(trainWhole)

    # split into X and Y
    if appendClass:
        trainX = trainWhole
    else:
        trainX = trainWhole[:,:-1]
    trainY = trainWhole[:,-1]    

    return trainX, trainY



def main():
    # Load both squares and spheres datasets
    img_dim = 10    # 2, 4, 10, 15, 28
    cubeDatasets = loadPickledData('../data/cubes/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/cubes/test_%d_50000.pkl.gz' % img_dim)
    sphereDatasets = loadPickledData('../data/spheres/train_%d_50000.pkl.gz' % img_dim,
                                     '../data/spheres/test_%d_50000.pkl.gz' % img_dim)

    # If necessary, reduce to 20000 rows to prevent memory error
    #cubeDatasets = ((cubeDatasets[0][0][:20000,:], None),
    #                (None, None),
    #                (cubeDatasets[2][0][:20000,:], None))
    #sphereDatasets = ((sphereDatasets[0][0][:20000,:], None),
    #                  (None, None),
    #                  (sphereDatasets[2][0][:20000,:], None))

    # make different size datasets
    sizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000]
    #sizes = [10, 20]

    sizedDatasetsX = {}
    sizedDatasetsXY = {}
    for size in sizes:
        sizedDatasetsX[size]  = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = False)
        sizedDatasetsXY[size] = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = True)
    testDatasetX  = makeSizedDataset(40000, cubeDatasets, sphereDatasets, appendClass = False)
    testDatasetXY = makeSizedDataset(40000, cubeDatasets, sphereDatasets, appendClass = True)

    print 'done loading.'

    for useXY in [False, True]:
        for size in sizes:
            print 'useXY', useXY, ', Size:', size
            thisDir = os.path.join(resman.rundir, '%s_size_%05d' % ('xy' if useXY else 'x', size))
            os.mkdir(thisDir)
            if useXY:
                thisDataset = (sizedDatasetsXY[size], (array([]), None), testDatasetXY)
            else:
                thisDataset = (sizedDatasetsX[size],  (array([]), None), testDatasetX)

            # this automatically saves the RBM to the given directory
            rbm, meanCosts = test_rbm(datasets = thisDataset,
                                      training_epochs = 45,
                                      img_dim = img_dim,
                                      n_hidden = 200, 
                                      learning_rate = .1, 
                                      output_dir = thisDir,
                                      quickHack = False,
                                      imgPlotFunction = lambda xx: xx[:,0:img_dim*img_dim],  # HACK: plot first slice
                                      )


if __name__ == '__main__':
    resman.start('junk', diary = True)
    main()
    resman.stop()
