#! /usr/bin/env ipython --gui=wx

import os, pdb, gzip, sys
import argparse
from numpy import mgrid, array, ones, zeros, linspace, random, reshape
from tvtk.api import tvtk

from utils import loadFromFile
from squaresRbm import loadPickledData
from GitResultsManager import resman
from util.plotting import plot3DShape



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
    nIter = 10 if mode == 'gibbs' else 200
    for ii in range(nIter):

        if mode == 'data':
            thisShape = xx[ii,:]
            plot3DShape(thisShape, Nw,
                        os.path.join(rundir, 'data_%03d.png' % ii),
                        smoothed = smoothed,
                        figSize = figSize)
        elif mode == 'filter':
            thisShape = rbm.W[:,ii]
            plot3DShape(thisShape, Nw,
                        os.path.join(rundir, 'filter_%03d.png' % ii),
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
    
