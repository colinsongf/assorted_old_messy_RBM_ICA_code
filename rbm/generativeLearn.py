#! /usr/bin/env python

import numpy, time, gzip, PIL.Image, os, pdb
import pickle, sys
from numpy import *

from ResultsManager import resman
from squaresRbm import loadPickledData
from rbm import RBM, test_rbm




def makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass):
    random.seed(0)

    #maxSize = cubeDatasets[0][0].shape[0]
    #print maxSize

    #sizedDatasets = {}
    #for size in sizes:

    # make pseudo dataset with squares vs. spheres labled
    trainX = vstack((cubeDatasets[0][0][0:size,:], sphereDatasets[0][0][0:size,:]))
    #testX  = vstack((cubeDatasets[2][0][:,:],      sphereDatasets[2][0][:,:]))

    trainX = array(trainX, dtype = bool)

    trainY   = array(vstack((zeros((size, 1)),    ones((size, 1)))), dtype = bool)
    #testY    = vstack((zeros((maxSize, 1)), ones((maxSize, 1))))

    trainX = hstack((trainX, trainY))
    #testX  = hstack((testX,  testY))

    # shuffle order of dataset
    random.shuffle(trainX)

    trainY = trainX[:,-1]
    
    if not appendClass:
        trainX = trainX[:,:-1]

    #random.shuffle(testX)

    #sizedDatasets[size] = ((trainX, trainY), (None, None), (testX, testY))

    return trainX, trainY



def main():
    
    # Load both squares and spheres datasets
    img_dim = 10    # 2, 4, 10, 15, 28
    cubeDatasets = loadPickledData('../data/cubes/train_%d_50000.pkl.gz' % img_dim,
                                   '../data/cubes/test_%d_50000.pkl.gz' % img_dim)
    sphereDatasets = loadPickledData('../data/spheres/train_%d_50000.pkl.gz' % img_dim,
                                     '../data/spheres/test_%d_50000.pkl.gz' % img_dim)

    #pdb.set_trace()
    #cubeX   =   cubeDatasets[0][0][:40000, :]
    #sphereX = sphereDatasets[0][0][:40000, :]

    # reduce to 20000 rows to prevent memory error
    #cubeDatasets = ((cubeDatasets[0][0][:20000,:], None),
    #                (None, None),
    #                (cubeDatasets[2][0][:20000,:], None))
    #sphereDatasets = ((sphereDatasets[0][0][:20000,:], None),
    #                  (None, None),
    #                  (sphereDatasets[2][0][:20000,:], None))

    # make different size datasets
    sizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000]
    #sizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000]
    #sizes = [10, 20]

    sizedDatasetsX = {}
    sizedDatasetsXY = {}
    for size in sizes:
        sizedDatasetsX[size]  = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = False)
        sizedDatasetsXY[size] = makeSizedDataset(size, cubeDatasets, sphereDatasets, appendClass = False)
    tesetDataset = makeSizedDataset(40000, cubeDatasets, sphereDatasets, appendClass = False)

    print 'GOOD'
    pdb.set_trace()

    xDatasets  = makeSizedDatasets(sizes, cubeDatasets, sphereDatasets, appendClass = False)
    xyDatasets = makeSizedDatasets(sizes, cubeDatasets, sphereDatasets, appendClass = True)
    #xDatasets  = makeSizedDatasets(sizes, cubeX, sphereX, appendClass = False)
    #xyDatasets = makeSizedDatasets(sizes, cubeX, sphereX, appendClass = True)
        
    
    # train something...
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


if __name__ == '__main__':
    resman.start('junk', diary = False)

    main()

    resman.stop()
