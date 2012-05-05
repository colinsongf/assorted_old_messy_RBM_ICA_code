#! /usr/bin/env ipythonwx

import os, pdb, gzip, sys
import argparse
from numpy import mgrid, array, ones
from tvtk.api import tvtk

from utils import loadFromFile
from squaresRbm import loadPickledData



def main(dataFilename, rbmFilename, smoothed = False):
    if dataFilename and rbmFilename:
        mode = 'sample'
    elif dataFilename:
        mode = 'data'
    else:
        mode = 'filter'
        
    from mayavi import mlab
    from mayavi.mlab import points3d, contour3d, plot3d


    if mode == 'data':
        xx, yy = loadPickledData(dataFilename)


        Nw = 10
        indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]

        cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])
        cubeEdges *= Nw

        for ii in range(10):
            fig = mlab.figure(0, size=(600,600))
            mlab.clf(fig)
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
            plot3d(cubeEdges[0,:], cubeEdges[1,:], cubeEdges[2,:], color=(.5,.5,.5),
                   line_width = 0,
                   representation = 'wireframe',
                   opacity = 1)
            #thisShape = rbm.W[:,ii]
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

            #mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
            mlab.view(24, 74, 33, (5, 5, 5))
            
            filename = 'test_%02d.png' % ii
            mlab.savefig(filename)    #, size=(800,800))

            if raw_input('Saved %s, enter to continue (q to quit)...' % filename) == 'q':
                return




    else:
        NOT_DONE_YET

    rbm = loadFromFile(rbmFilename)
    datasets = loadPickledData('../data/spheres/train_%d_50000.pkl.gz' % img_dim,
                               '../data/spheres/test_%d_50000.pkl.gz' % img_dim)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize input data, RBM filters, or RBM samples.')
    parser.add_argument('--data', metavar = 'data_filename', type = str, nargs = 1,
                        help='Data filename to load')
    parser.add_argument('--rbm', metavar = 'rbm_filename', type = str, nargs = 1,
                        help='RBM to load from .pkl.gz file')
    args = parser.parse_args()

    if not args.data and not args.rbm:
        parser.error('Must specify --data or --rbm.')

    main(args.data[0] if args.data else None,
         args.rbm[0]  if args.rbm  else None)
