#! /usr/bin/env python

import os, sys
import pdb
import Image
from numpy import *

from matplotlib import pyplot

from util.dataLoaders import saveToFile
from util.plotting import tile_raster_images



def paintSquare(indexI, indexJ, startI, endI, startJ, endJ):
    arr = ones(indexI.shape, dtype='bool')
    arr &= (indexI >= startI)
    arr &= (indexJ >= startJ)
    arr &= (indexI < endI)
    arr &= (indexJ < endJ)
    return arr
    


def paintCircle(indexI, indexJ, locI, locJ, radius):
    return array(((indexI-locI)**2 + (indexJ-locJ)**2) < radius**2, dtype = float32)



def randomSquareCircle(Nw = 10, Nsamples = 10, prob = .5):
    if Nw % 2 != 0:
        raise Exception('Nw must be divisible by 2.')
    halfsize = Nw/2
    
    retX = zeros((Nsamples, Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 2), dtype = float32)

    # compute once to save time
    indexJ, indexI = meshgrid(range(halfsize), range(halfsize))

    for ii in xrange(Nsamples):
        selectSquare = random.rand() < prob
        selectCircle = random.rand() < prob

        if selectSquare:
            square = paintSquare(indexI, indexJ, 0, halfsize, 0, halfsize)
            # select bottom left corner
            reshape(retX[ii,:], (Nw,Nw))[halfsize:Nw, 0:halfsize] = square
        if selectCircle:
            circle = paintCircle(indexI, indexJ, halfsize/2, halfsize/2, halfsize/2 - .75)
            # select top right corner
            reshape(retX[ii,:], (Nw,Nw))[0:halfsize, halfsize:Nw] = circle
        
        retY[ii,:] = [selectSquare, selectCircle]

    return retX, retY



def random4SquaresCircles(Nw = 10, Nsamples = 10, num = 10, prob = .5):
    if Nw % 2 != 0:
        raise Exception('Nw must be divisible by 2.')
    halfsize = Nw/2
    
    retX = zeros((Nsamples, Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 8), dtype = float32)

    # compute once to save time
    indexJ, indexI = meshgrid(range(halfsize), range(halfsize))

    for ii in xrange(Nsamples):
        # randomly pick coordinates of square
        for corner in range(4):
            selectCorner = random.rand() < prob
            if selectCorner:
                isCircle = random.rand() < .5
                retY[ii,selectCorner + 4*isCircle] = 1
                whichI = corner / 2
                whichJ = corner % 2
                if not isCircle:
                    square = paintSquare(indexI, indexJ, 0, halfsize, 0, halfsize)
                    reshape(retX[ii,:], (Nw,Nw))[(whichI*halfsize):((whichI+1)*halfsize),
                                                 (whichJ*halfsize):((whichJ+1)*halfsize)] = square                    
                else:
                    circle = paintCircle(indexI, indexJ, halfsize/2, halfsize/2, halfsize/2 - .75)
                    reshape(retX[ii,:], (Nw,Nw))[(whichI*halfsize):((whichI+1)*halfsize),
                                                 (whichJ*halfsize):((whichJ+1)*halfsize)] = circle

    return retX, retY



def demo():
    random.seed(0)
    Nw = 2
    xx, yy = randomSquareCircle(Nw = Nw, Nsamples = 150)

    image = Image.fromarray(tile_raster_images(
        X = xx, img_shape = (Nw,Nw),
        tile_shape = (10, 15), tile_spacing=(1,1),
        scale_rows_to_unit_interval = False))
    image.save('demo.png')
    print 'saved as demo.png'
    image.show()



def makeData():
    for Nw in (2, 4, 10, 16, 28, 50):
        #for Nsamples in (50, 500, 5000):
        for Nsamples in (50, 500, 5000, 50000):
            random.seed(0)
            for string in ['train_sc_p5', 'test_sc_p5', 'train_sc_p1', 'test_sc_p1',
                           'train_4sc_p5', 'test_4sc_p5', 'train_4sc_p1', 'test_4sc_p1']:

                probability = .5 if 'p5' in string else .1
                function = random4SquaresCircles if '4sc' in string else randomSquareCircle

                sampleXAndY = function(Nw, Nsamples, prob = probability)
                saveToFile('../data/simpleShapes/%s_%02d_%d.pkl.gz' % (string, Nw, Nsamples), sampleXAndY)
                xx, yy = sampleXAndY
                if Nsamples == 5000:
                    image = Image.fromarray(tile_raster_images(
                        X = xx, img_shape = (Nw,Nw),
                        tile_shape = (10, 15), tile_spacing=(1,1),
                        scale_rows_to_unit_interval = False))
                    image.save('../data/simpleShapes/%s_%02d.png' % (string, Nw))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Running demo. Run as "%s make" to generate and save data.'
        demo()
    else:
        makeData()
