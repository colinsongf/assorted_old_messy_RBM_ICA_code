#! /usr/local/bin/ipython --gui=wx


import os, pdb, gzip, sys
from PIL import Image
from numpy import *
import cPickle as pickle

from matplotlib import pyplot
from rbm.utils import imagesc
from makeUpsonRovio1 import saveToFile



def paintCube(indexX, indexY, indexZ, bounds):
    '''bounds is 2x3'''

    xx = logical_and(indexX >= bounds[0,0], indexX <= bounds[1,0])
    yy = logical_and(indexY >= bounds[0,1], indexY <= bounds[1,1])
    zz = logical_and(indexZ >= bounds[0,2], indexZ <= bounds[1,2])
    
    return array(logical_and(logical_and(xx, yy), zz), dtype = float32)



def randomCubeSampleMatrix(Nw = 10, Nsamples = 10, minSize = 2):
    retX = zeros((Nsamples, Nw * Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 6), dtype = float32)

    # compute once to save time
    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]
    
    for ii in xrange(Nsamples):
        # randomly pick coordinates of cube
        tries = 0
        bounds = random.randint(0, Nw, (2,3))
        while not all(bounds[0,:] - bounds[1,:] >= minSize):
            bounds = random.randint(0, Nw, (2,3))
            if tries > 100:
                raise Exception('failed too many times')

        # sort bounds in (min, max) order
        bounds.sort(0)

        thisImg = paintCube(indexX, indexY, indexZ, bounds)

        retX[ii,:] = thisImg.flatten()
        retY[ii,:] = bounds.flatten()
    
    return retX, retY




def demo(smoothed = False):
    from mayavi import mlab
    from mayavi.mlab import points3d, contour3d, plot3d
    
    random.seed(0)
    Nw = 15
    xx, yy = randomCubeSampleMatrix(Nw = Nw, Nsamples = 50)

    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]

    cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])
    cubeEdges *= Nw

    for ii in range(10):
        fig = mlab.gcf()
        #fig.scene.disable_render = True

        mlab.clf(fig)
        #from tvtk.api import tvtk
        #fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        #mlab.clf()
        plot3d(cubeEdges[0,:], cubeEdges[1,:], cubeEdges[2,:], color=(.5,.5,.5),
               line_width = 0,
               representation = 'wireframe',
               opacity = 1)
        thisShape = xx[ii,:]
        if smoothed:
            contour3d(reshape(thisShape, (Nw,Nw,Nw)), contours=[.5], color=(1,1,1))
        else:
            points3d(indexX.flatten()[thisShape > 0],
                     indexY.flatten()[thisShape > 0],
                     indexZ.flatten()[thisShape > 0],
                     ones(sum(thisShape > 0)) * .9,
                     color = (1,1,1),
                     mode = 'cube',
                     scale_factor = 1.0)

        mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view

        filename = 'test_%02d.png' % ii
        mlab.savefig(filename)    #, size=(800,800))

        if raw_input('Saved %s, enter to continue (q to quit)...' % filename) == 'q':
            return
    

def main():
    random.seed(0)

    #saveToFile('../data/cubes/train_02_50.pkl.gz',    randomCubeSampleMatrix(Nw = 02, Nsamples = 50))
    #saveToFile('../data/cubes/test_02_50.pkl.gz',     randomCubeSampleMatrix(Nw = 02, Nsamples = 50))
    #saveToFile('../data/cubes/train_02_50000.pkl.gz', randomCubeSampleMatrix(Nw = 02, Nsamples = 50000))
    #saveToFile('../data/cubes/test_02_50000.pkl.gz',  randomCubeSampleMatrix(Nw = 02, Nsamples = 50000))

    saveToFile('../data/cubes/train_04_50.pkl.gz',    randomCubeSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/cubes/test_04_50.pkl.gz',     randomCubeSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/cubes/train_04_50000.pkl.gz', randomCubeSampleMatrix(Nw = 04, Nsamples = 50000))
    saveToFile('../data/cubes/test_04_50000.pkl.gz',  randomCubeSampleMatrix(Nw = 04, Nsamples = 50000))

    saveToFile('../data/cubes/train_10_50.pkl.gz',    randomCubeSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/cubes/test_10_50.pkl.gz',     randomCubeSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/cubes/train_10_50000.pkl.gz', randomCubeSampleMatrix(Nw = 10, Nsamples = 50000))
    saveToFile('../data/cubes/test_10_50000.pkl.gz',  randomCubeSampleMatrix(Nw = 10, Nsamples = 50000))

    saveToFile('../data/cubes/train_15_50.pkl.gz',    randomCubeSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/cubes/test_15_50.pkl.gz',     randomCubeSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/cubes/train_15_50000.pkl.gz', randomCubeSampleMatrix(Nw = 15, Nsamples = 50000))
    saveToFile('../data/cubes/test_15_50000.pkl.gz',  randomCubeSampleMatrix(Nw = 15, Nsamples = 50000))

    # too big! Fails with ValueError: array is too big.
    #saveToFile('../data/cubes/train_28_50.pkl.gz',    randomCubeSampleMatrix(Nw = 28, Nsamples = 50))
    #saveToFile('../data/cubes/test_28_50.pkl.gz',     randomCubeSampleMatrix(Nw = 28, Nsamples = 50))
    #saveToFile('../data/cubes/train_28_50000.pkl.gz', randomCubeSampleMatrix(Nw = 28, Nsamples = 50000))
    #saveToFile('../data/cubes/test_28_50000.pkl.gz',  randomCubeSampleMatrix(Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in ('demo', 'data'):
        print 'Usage: %s demo    # run a demo\n       %s data    # create and save cubes data.' % (sys.argv[0], sys.argv[0])
        sys.exit(1)

    if sys.argv[1] == 'demo':
        demo()
    else:
        main()

