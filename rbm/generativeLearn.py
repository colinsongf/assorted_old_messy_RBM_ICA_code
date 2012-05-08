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

    # reduce to 20000 rows to prevent memory error
    #cubeDatasets = ((cubeDatasets[0][0][:20000,:], None),
    #                (None, None),
    #                (cubeDatasets[2][0][:20000,:], None))
    #sphereDatasets = ((sphereDatasets[0][0][:20000,:], None),
    #                  (None, None),
    #                  (sphereDatasets[2][0][:20000,:], None))

    # make different size datasets
    #sizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000]
    sizes = [10, 20]

    sizedDatasetsX = {}
    sizedDatasetsXY = {}
    for size in sizes:
        sizedDatasetsX[size]  = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = False)
        sizedDatasetsXY[size] = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = False)
    testDataset = makeSizedDataset(40000, cubeDatasets, sphereDatasets, appendClass = False)

    #xDatasets  = makeSizedDatasets(sizes, cubeDatasets, sphereDatasets, appendClass = False)
    #xyDatasets = makeSizedDatasets(sizes, cubeDatasets, sphereDatasets, appendClass = True)
    #xDatasets  = makeSizedDatasets(sizes, cubeX, sphereX, appendClass = False)
    #xyDatasets = makeSizedDatasets(sizes, cubeX, sphereX, appendClass = True)
    
    # train something...
    print 'done loading.'

    for size in sizes:
        print 'Size:', size
        thisDir = os.path.join(resman.rundir, 'size_%05d' % size)
        os.mkdir(thisDir)
        thisDataset = (sizedDatasetsX[size], (array([]), None), testDataset)
        rbm, meanCosts = test_rbm(datasets = thisDataset,
                                  training_epochs = 1,
                                  img_dim = img_dim,
                                  n_hidden = 200, 
                                  learning_rate = .1, 
                                  output_dir = thisDir,
                                  quickHack = False,
                                  imgPlotFunction = lambda xx: xx[:,0:img_dim*img_dim],  # HACK: plot first slice
                                  )


if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
