#! /usr/bin/env ipythonwx

import os, pdb, gzip, sys
import argparse
from numpy import mgrid, array, ones, zeros, linspace, random
from tvtk.api import tvtk

from utils import loadFromFile
from squaresRbm import loadPickledData
from ResultsManager import resman


from mayavi import mlab
from mayavi.mlab import points3d, contour3d, plot3d

cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])


def plot3DShape(shape, Nw,
                saveFilename = None, smoothed = False, visSimple = True,
                plotThresh = 0, figSize = (300,300)):
    '''Plots a 3D shape of size Nw x Nw x Nw inside a frame.'''

    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]
    edges = cubeEdges * Nw
    
    fig = mlab.figure(0, size = figSize)
    mlab.clf(fig)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    plot3d(edges[0,:], edges[1,:], edges[2,:], color=(.5,.5,.5),
           line_width = 0,
           representation = 'wireframe',
           opacity = 1)
    
    if smoothed:
        contour3d(reshape(shape, (Nw,Nw,Nw)), contours=[.5], color=(1,1,1))
    else:
        print saveFilename
        mn = shape.min()
        mx = shape.max()
        idx = (shape > plotThresh)
        print mn, mx, sum(idx)
        #pdb.set_trace()
        if sum(idx) > 0:
            if visSimple:
                pts = points3d(indexX.flatten()[idx] + .5,
                               indexY.flatten()[idx] + .5,
                               indexZ.flatten()[idx] + .5,
                               ones(sum(idx)) * .9,
                               #((shape-mn) / (mx-mn) * .9)[idx],
                               color = (1,1,1),
                               mode = 'cube',
                               scale_factor = 1.0)
            else:
                pts = points3d(indexX.flatten()[idx] + .5,
                         indexY.flatten()[idx] + .5,
                         indexZ.flatten()[idx] + .5,
                         #ones(sum(idx)) * .9,
                         ((shape-mn) / (mx-mn) * .9)[idx],
                         colormap = 'bone',
                         #color = (1,1,1),
                         mode = 'cube',
                         scale_factor = 1.0)
            lut = pts.module_manager.scalar_lut_manager.lut.table.to_array()
            tt = linspace(0, 255, 256)
            lut[:, 0] = tt*0 + 255
            lut[:, 1] = tt*0 + 255
            lut[:, 2] = tt*0 + 255
            lut[:, 3] = tt
            pts.module_manager.scalar_lut_manager.lut.table = lut

    #mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
    mlab.view(24, 74, 33, (5, 5, 5))

    mlab.draw()

    if saveFilename:
        mlab.savefig(saveFilename)



def main(dataFilename, rbmFilename,  rundir, smoothed = False, plotEvery = 1):
    random.seed(0)
    
    if dataFilename:
        mode = 'data'
        xx, yy = loadPickledData(dataFilename)
        Nw = int(round(xx.shape[1]**.33333333))

    if rbmFilename:
        mode = 'filter'
        rbm = loadFromFile(rbmFilename)
        Nw = int(round(rbm.W.shape[0]**.33333333))

    if dataFilename and rbmFilename:
        mode = 'gibbs'   # override
        

    # Setup
    figSize = (800,800)
    
    # Main plotting loop
    for ii in range(10):

        if mode == 'data':
            thisShape = xx[ii,:]
            plot3DShape(thisShape, Nw,
                        os.path.join(rundir, 'data_%02d.png' % ii),
                        smoothed = smoothed,
                        figSize = figSize)
        elif mode == 'filter':
            thisShape = rbm.W[:,ii]
            plot3DShape(thisShape, Nw,
                        os.path.join(rundir, 'filter_%02d.png' % ii),
                        smoothed = smoothed,
                        visSimple = False,
                        figSize = figSize)
        else:
            # gibbs
            idx = random.randint(0, xx.shape[1])

            nSamples = 200
            samples = zeros((nSamples, Nw**3))

            visMean = xx[idx,:]
            visSample = visMean
            for jj in xrange(nSamples):
                samples[jj,:] = visMean # show the mean, but use the sample for gibbs steps
                for ss in xrange(plotEvery):
                    visMean, visSample = rbm.gibbs_vhv(visSample)[4:6]   # 4 for mean, 5 for sample

            print ' ... plotting sample ', ii
            for jj in xrange(nSamples):
                plot3DShape(samples[jj,:], Nw,
                            os.path.join(rundir, 'sample_%02d_%05d.png' % (ii, jj)),
                            smoothed = smoothed,
                            plotThresh = .5,
                            figSize = figSize)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize input data, RBM filters, or RBM samples.')
    parser.add_argument('--name', metavar = 'name', type = str,
                        default = 'junk',
                        help='Name of run for ResultsManager. Default: junk')
    parser.add_argument('--data', metavar = 'filename', type = str,
                        help='Data filename to load')
    parser.add_argument('--rbm', metavar = 'filename', type = str,
                        help='RBM to load from .pkl.gz file')
    parser.add_argument('--plotEvery', metavar = 'steps', type = int,
                        default = 1,
                        help='How many Gibbs sampling steps to take between each plot. Default: 1')
    args = parser.parse_args()

    if not args.data and not args.rbm:
        parser.error('Must specify --data or --rbm.')

    resman.start(args.name, diary = True)
    
    main(args.data if args.data else None,
         args.rbm  if args.rbm  else None,
         rundir = resman.rundir,
         plotEvery = args.plotEvery)

    resman.stop()
    
