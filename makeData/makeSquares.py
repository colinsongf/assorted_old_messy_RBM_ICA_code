#! /usr/bin/env python


import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle

from matplotlib import pyplot
from rbm.utils import imagesc
from makeUpsonRovio1 import saveToFile



def paintSquare(img, startI, startJ, sizeI, sizeJ):
    # trim coords if necessary
    img[max(0, startI):startI + sizeI,
        max(0, startJ):startJ + sizeJ] = 1
    return img
    


def randomSquareSampleMatrix(Nw = 10, Nsamples = 10):
    ret = zeros((Nsamples, Nw * Nw), dtype = float32)
    
    for ii in xrange(Nsamples):
        # randomly pick coordinates of square
        sizeI  = random.randint(1, Nw/2 + 1)
        sizeJ  = random.randint(1, Nw/2 + 1)
        startI = random.randint(-sizeI + 1, Nw)
        startJ = random.randint(-sizeJ + 1, Nw)

        thisImg = paintSquare(reshape(ret[ii,:], (Nw,Nw)), startI, startJ, sizeI, sizeJ)

        ret[ii,:] = thisImg.flatten()
        
    return ret



def main(demo = True):
    if demo:
        random.seed(0)
        foo = randomSquareSampleMatrix(Nw = 15, Nsamples = 25)

        pyplot.figure()
        for ii in range(25):
            ax = pyplot.subplot(5,5,ii)
            imagesc(foo[ii,:].reshape((15,15)), ax=ax)
        pyplot.show()

    random.seed(0)

    saveToFile('../data/squares/train_02_50.pkl.gz',    randomSquareSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/squares/test_02_50.pkl.gz',     randomSquareSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/squares/train_02_50000.pkl.gz', randomSquareSampleMatrix(Nw = 02, Nsamples = 50000))
    saveToFile('../data/squares/test_02_50000.pkl.gz',  randomSquareSampleMatrix(Nw = 02, Nsamples = 50000))

    saveToFile('../data/squares/train_04_50.pkl.gz',    randomSquareSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/squares/test_04_50.pkl.gz',     randomSquareSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/squares/train_04_50000.pkl.gz', randomSquareSampleMatrix(Nw = 04, Nsamples = 50000))
    saveToFile('../data/squares/test_04_50000.pkl.gz',  randomSquareSampleMatrix(Nw = 04, Nsamples = 50000))

    saveToFile('../data/squares/train_10_50.pkl.gz',    randomSquareSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/squares/test_10_50.pkl.gz',     randomSquareSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/squares/train_10_50000.pkl.gz', randomSquareSampleMatrix(Nw = 10, Nsamples = 50000))
    saveToFile('../data/squares/test_10_50000.pkl.gz',  randomSquareSampleMatrix(Nw = 10, Nsamples = 50000))

    saveToFile('../data/squares/train_15_50.pkl.gz',    randomSquareSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/squares/test_15_50.pkl.gz',     randomSquareSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/squares/train_15_50000.pkl.gz', randomSquareSampleMatrix(Nw = 15, Nsamples = 50000))
    saveToFile('../data/squares/test_15_50000.pkl.gz',  randomSquareSampleMatrix(Nw = 15, Nsamples = 50000))
    
    saveToFile('../data/squares/train_28_50.pkl.gz',    randomSquareSampleMatrix(Nw = 28, Nsamples = 50))
    saveToFile('../data/squares/test_28_50.pkl.gz',     randomSquareSampleMatrix(Nw = 28, Nsamples = 50))
    saveToFile('../data/squares/train_28_50000.pkl.gz', randomSquareSampleMatrix(Nw = 28, Nsamples = 50000))
    saveToFile('../data/squares/test_28_50000.pkl.gz',  randomSquareSampleMatrix(Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    main(demo = False)
