#! /usr/bin/env ipythonwx

import os, pdb, gzip, sys
import argparse
from numpy import mgrid, array, ones, linspace
from tvtk.api import tvtk

from utils import loadFromFile
from squaresRbm import loadPickledData



cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])



def main(dataFilename, rbmFilename, smoothed = False):
    from mayavi import mlab
    from mayavi.mlab import points3d, contour3d, plot3d

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
    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]
    edges = cubeEdges * Nw
    figSize = (300,300)
    
    # Main plotting loop
    for ii in range(10):
        fig = mlab.figure(0, size = figSize)
        mlab.clf(fig)
        fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        plot3d(edges[0,:], edges[1,:], edges[2,:], color=(.5,.5,.5),
               line_width = 0,
               representation = 'wireframe',
               opacity = 1)

        if mode == 'data':
            thisShape = xx[ii,:]
        elif mode == 'filter':
            thisShape = rbm.W[:,ii]
        else:
            NOTDONEYET

        if smoothed:
            contour3d(reshape(thisShape, (Nw,Nw,Nw)), contours=[.5], color=(1,1,1))
        else:
            #pdb.set_trace()
            mn = thisShape.min()
            mx = thisShape.max()
            idx = (thisShape > 0)
            pts = points3d(indexX.flatten()[idx] + .5,
                     indexY.flatten()[idx] + .5,
                     indexZ.flatten()[idx] + .5,
                     #ones(sum(idx)) * .9,
                     ((thisShape-mn) / (mx-mn) * .9)[idx],
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
            mlab.draw()

        #mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
        mlab.view(24, 74, 33, (5, 5, 5))

        #pdb.set_trace()

        if raw_input('(q to quit)...') == 'q':
            return

        filename = 'demo_%02d.png' % ii
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
