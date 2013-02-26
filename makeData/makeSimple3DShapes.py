#! /usr/bin/env ipythonpl

import os, sys
import pdb
from numpy import *

from util.dataLoaders import saveToFile
from util.plotting import plot3DShape

from makeCubes import paintCube



def paintSphere(indexX, indexY, indexZ, loc, radius):
    return array(((indexX-loc[0])**2 + (indexY-loc[1])**2 + (indexZ-loc[2])**2) < radius**2)



def randomCubeSphere(size, Nsamples = 10, minRadius = .5):
    Nx,Ny,Nz = size

    retX = zeros((Nsamples, Nx*Ny*Nz), dtype = float32)

    # compute once to save time
    indexX, indexY, indexZ = meshgrid(range(Nx),range(Ny),range(Nz))
    
    for ii in xrange(Nsamples):
        thisShape = zeros(size, dtype='bool')

        # random number of cubes and spheres
        Ncubes   = random.poisson(1)
        Nspheres = random.poisson(1)

        for cc in range(Ncubes):
            xmm = sorted(random.randint(0,Nx,2))
            ymm = sorted(random.randint(0,Ny,2))
            zmm = sorted(random.randint(0,Nz,2))
            thisShape |= paintCube(indexX, indexY, indexZ,
                                   array([xmm, ymm, zmm]).T, asBool = True)

        for ss in range(Nspheres):
            loc = random.rand(3) * size
            radius = random.uniform(minRadius, min(size)/3.0)
            thisShape |= paintSphere(indexX, indexY, indexZ, loc, radius)

        retX[ii,:] = thisShape.flatten()
        #print ii, thisShape.min(), 'to', thisShape.max(), 'sum is', thisShape.sum()

    # Convert to -1 and +1
    retX -= .5
    retX *= 2.0

    return retX



def demo():
    random.seed(0)
    size = (10,10,20)

    xx = randomCubeSphere(size, Nsamples = 50)

    for ii in range(10):
        plot3DShape(reshape(xx[ii,:], size), figSize = (800,800))
        raw_input('push enter for next')



def makeData():
    size = (10,10,20)

    for Nsamples in (50, 500, 5000, 50000):
        for seed,string in ((0, 'train'), (1, 'test')):
            random.seed(seed)

            xx = randomCubeSphere(size, Nsamples = Nsamples)
            saveToFile('../data/simple3DShapes/poisson_%s_%d.pkl.gz' % (string, Nsamples), xx)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Running demo. Run as "%s make" to generate and save data.' % sys.argv[0]
        demo()
    else:
        makeData()
