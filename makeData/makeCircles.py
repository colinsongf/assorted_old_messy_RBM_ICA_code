#! /usr/bin/env python


import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle
import numpy

from matplotlib import pyplot
from util.plotting import tile_raster_images
from rbm.utils import imagesc
from makeUpsonRovio1 import saveToFile



def paintCircle(img, indexI, indexJ, locI, locJ, radius):
    return array(((indexI-locI)**2 + (indexJ-locJ)**2) < radius**2, dtype = float32)



def randomCircleSampleMatrix(Nw = 10, Nsamples = 10):
    retX = zeros((Nsamples, Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 3), dtype = float32)

    # compute once to save time
    indexJ, indexI = meshgrid(range(Nw), range(Nw))
    
    for ii in xrange(Nsamples):
        # randomly pick coordinates of circle
        locI   = random.randint(0, Nw)
        locJ   = random.randint(0, Nw)
        radius = random.uniform(1, float(Nw)/3)  # biggest circle is 2/3 width of box

        thisImg = paintCircle(reshape(retX[ii,:], (Nw,Nw)), indexI, indexJ, locI, locJ, radius)

        retX[ii,:] = thisImg.flatten()
        retY[ii,:] = [locI, locJ, radius]
    
    return retX, retY



def demo():
    random.seed(0)
    Nw = 15
    xx, yy = randomCircleSampleMatrix(Nw = Nw, Nsamples = 100)


    saveImage = False
    if saveImage:
        image_data = numpy.ones(((Nw+1)*10-1,(Nw+1)*10-1), dtype='uint8') * 51  # dark gray
        tmp = tile_raster_images(X = xx,
                                 img_shape = (Nw,Nw),
                                 tile_shape = (10,10),
                                 tile_spacing = (1,1))
        image_data = tmp
        image = Image.fromarray(image_data)
        image.save(os.path.join('demoCircleSamples.png'))
    else:
        pyplot.figure()
        for ii in range(25):
            ax = pyplot.subplot(5,5,ii)
            imagesc(xx[ii,:].reshape((Nw,Nw)), ax=ax)
            #pyplot.title(repr(yy[ii,:]))
            pyplot.title('%d, %d, %.2f' % tuple(yy[ii,:]))
        pyplot.show()




def main():
    random.seed(0)

    saveToFile('../data/circles/train_02_50.pkl.gz',    randomCircleSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/circles/test_02_50.pkl.gz',     randomCircleSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/circles/train_02_50000.pkl.gz', randomCircleSampleMatrix(Nw = 02, Nsamples = 50000))
    saveToFile('../data/circles/test_02_50000.pkl.gz',  randomCircleSampleMatrix(Nw = 02, Nsamples = 50000))

    saveToFile('../data/circles/train_04_50.pkl.gz',    randomCircleSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/circles/test_04_50.pkl.gz',     randomCircleSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/circles/train_04_50000.pkl.gz', randomCircleSampleMatrix(Nw = 04, Nsamples = 50000))
    saveToFile('../data/circles/test_04_50000.pkl.gz',  randomCircleSampleMatrix(Nw = 04, Nsamples = 50000))

    saveToFile('../data/circles/train_10_50.pkl.gz',    randomCircleSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/circles/test_10_50.pkl.gz',     randomCircleSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/circles/train_10_50000.pkl.gz', randomCircleSampleMatrix(Nw = 10, Nsamples = 50000))
    saveToFile('../data/circles/test_10_50000.pkl.gz',  randomCircleSampleMatrix(Nw = 10, Nsamples = 50000))

    saveToFile('../data/circles/train_15_50.pkl.gz',    randomCircleSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/circles/test_15_50.pkl.gz',     randomCircleSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/circles/train_15_50000.pkl.gz', randomCircleSampleMatrix(Nw = 15, Nsamples = 50000))
    saveToFile('../data/circles/test_15_50000.pkl.gz',  randomCircleSampleMatrix(Nw = 15, Nsamples = 50000))
    
    saveToFile('../data/circles/train_28_50.pkl.gz',    randomCircleSampleMatrix(Nw = 28, Nsamples = 50))
    saveToFile('../data/circles/test_28_50.pkl.gz',     randomCircleSampleMatrix(Nw = 28, Nsamples = 50))
    saveToFile('../data/circles/train_28_50000.pkl.gz', randomCircleSampleMatrix(Nw = 28, Nsamples = 50000))
    saveToFile('../data/circles/test_28_50000.pkl.gz',  randomCircleSampleMatrix(Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    demo()
    #main()

