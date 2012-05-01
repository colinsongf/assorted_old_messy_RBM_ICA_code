#! /usr/local/bin/ipython --gui=wx


import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle

from matplotlib import pyplot
from rbm.utils import imagesc
from makeUpsonRovio1 import saveToFile



def paintSphere(img, indexX, indexY, indexZ, locX, locY, locZ, radius):
    return array(((indexX-locX)**2 + (indexY-locY)**2 + (indexZ-locZ)**2) < radius**2, dtype = float32)



def randomSphereSampleMatrix(Nw = 10, Nsamples = 10):
    retX = zeros((Nsamples, Nw * Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 4), dtype = float32)

    # compute once to save time
    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]
    
    for ii in xrange(Nsamples):
        # randomly pick coordinates of sphere
        locX   = random.randint(0, Nw)
        locY   = random.randint(0, Nw)
        locZ   = random.randint(0, Nw)
        radius = random.uniform(1, float(Nw)/3)  # biggest sphere is 2/3 width of box

        thisImg = paintSphere(reshape(retX[ii,:], (Nw,Nw,Nw)), indexX, indexY, indexZ, locX, locY, locZ, radius)

        retX[ii,:] = thisImg.flatten()
        retY[ii,:] = [locX, locY, locZ, radius]
    
    return retX, retY




def demo(smoothed = False):
    from mayavi import mlab
    from mayavi.mlab import points3d, contour3d, plot3d
    
    random.seed(0)
    Nw = 15
    xx, yy = randomSphereSampleMatrix(Nw = Nw, Nsamples = 50)

    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]

    cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])
    cubeEdges *= Nw
    plot3d(cubeEdges[0,:], cubeEdges[1,:], cubeEdges[2,:], color=(.5,.5,.5),
           line_width = 0,
           representation = 'wireframe',
           opacity = 1)

    mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
    for ii in range(10):
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

        fig = mlab.gcf()
        from tvtk.api import tvtk
        fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        mlab.show()
        #pdb.set_trace()
        raw_input('enter to continue...')
    

def main():
    random.seed(0)

    saveToFile('../data/spheres/train_02_50.pkl.gz',    randomSphereSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/spheres/test_02_50.pkl.gz',     randomSphereSampleMatrix(Nw = 02, Nsamples = 50))
    saveToFile('../data/spheres/train_02_50000.pkl.gz', randomSphereSampleMatrix(Nw = 02, Nsamples = 50000))
    saveToFile('../data/spheres/test_02_50000.pkl.gz',  randomSphereSampleMatrix(Nw = 02, Nsamples = 50000))

    saveToFile('../data/spheres/train_04_50.pkl.gz',    randomSphereSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/spheres/test_04_50.pkl.gz',     randomSphereSampleMatrix(Nw = 04, Nsamples = 50))
    saveToFile('../data/spheres/train_04_50000.pkl.gz', randomSphereSampleMatrix(Nw = 04, Nsamples = 50000))
    saveToFile('../data/spheres/test_04_50000.pkl.gz',  randomSphereSampleMatrix(Nw = 04, Nsamples = 50000))

    saveToFile('../data/spheres/train_10_50.pkl.gz',    randomSphereSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/spheres/test_10_50.pkl.gz',     randomSphereSampleMatrix(Nw = 10, Nsamples = 50))
    saveToFile('../data/spheres/train_10_50000.pkl.gz', randomSphereSampleMatrix(Nw = 10, Nsamples = 50000))
    saveToFile('../data/spheres/test_10_50000.pkl.gz',  randomSphereSampleMatrix(Nw = 10, Nsamples = 50000))

    saveToFile('../data/spheres/train_15_50.pkl.gz',    randomSphereSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/spheres/test_15_50.pkl.gz',     randomSphereSampleMatrix(Nw = 15, Nsamples = 50))
    saveToFile('../data/spheres/train_15_50000.pkl.gz', randomSphereSampleMatrix(Nw = 15, Nsamples = 50000))
    saveToFile('../data/spheres/test_15_50000.pkl.gz',  randomSphereSampleMatrix(Nw = 15, Nsamples = 50000))

    # too big! Fails with ValueError: array is too big.
    #saveToFile('../data/spheres/train_28_50.pkl.gz',    randomSphereSampleMatrix(Nw = 28, Nsamples = 50))
    #saveToFile('../data/spheres/test_28_50.pkl.gz',     randomSphereSampleMatrix(Nw = 28, Nsamples = 50))
    #saveToFile('../data/spheres/train_28_50000.pkl.gz', randomSphereSampleMatrix(Nw = 28, Nsamples = 50000))
    #saveToFile('../data/spheres/test_28_50000.pkl.gz',  randomSphereSampleMatrix(Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    #demo()
    main()
