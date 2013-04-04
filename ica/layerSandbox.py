#! /usr/bin/env ipython

'''
Research code

Jason Yosinski
'''

import ipdb as pdb
import os, sys, time
from numpy import *
from PIL import Image
from GitResultsManager import resman
from IPython.parallel import Client

from tica import neighborMatrix
#from visualize import plotImageData, plotCov, plotImageRicaWW, plotRicaActivations, plotRicaReconstructions
from util.dataLoaders import loadAtariData, loadUpsonData, loadRandomData, saveToFile, loadFromPklGz
#from util.dataPrep import PCAWhiteningDataNormalizer, printDataStats
#from util.misc import pt, pc
from makeData.makeUpsonRovio2 import randomSampleMatrix, trainFilter, testFilter


@profile
def main():
    #resman.start('junk', diary = False)

    tic = time.time()
    
    tica  = loadFromPklGz('results/130402_033310_44cc757_master_psearchTica_UP/00022_data/tica.pkl.gz')
    myres = loadFromPklGz('results/130402_033310_44cc757_master_psearchTica_UP/00022_data/myresults.pkl.gz')
    # myres['params']['dataPath'] is '../data/upson_rovio_2/train_10_50000_1c.pkl.gz'
    whitener = loadFromPklGz('../data/upson_rovio_2/white/train_10_50000_1c.whitener.pkl.gz')

    seed = 0
    isColor = False
    Nw = 10
    Nw5  = Nw/2
    Nw15 = int(Nw * 1.5)
    Nsamples = 1000

    nColors = 3 if isColor else 1

    if Nw%2 != 0:
        raise Exception('only works for even patch sizes')
    if isColor:
        raise Exception('no colors yet....')
    if myres['params']['hiddenISize'] != myres['params']['hiddenJSize']:
        raise Exception('representation embedding must be square (for now)')
    
    # Sample 1.5x larger windows (e.g. 15 px on a side). This is for
    # patches that overlap half with their
    # neighbors. largeSamples.shape is (Nsamples, Nw15^2)
    largeSamples = randomSampleMatrix(trainFilter, seed, isColor, Nw = Nw15, Nsamples = Nsamples)

    # Sample each corner of the 1.5x patch (e.g. 4 patches 10px on a
    # side). Color order is [ii_r ii_g ii_b ii+1_r ii+1_g ii+1_b ...]
    stackedSmallSamples = zeros((4*Nsamples, Nw*Nw))
    counter = 0
    for largeIdx in xrange(Nsamples):
        largePatch = reshape(largeSamples[largeIdx,:], (Nw15,Nw15))
        for ii in (0,1):
            for jj in (0,1):
                stackedSmallSamples[counter,:] = largePatch[(ii*Nw5):(ii*Nw5+Nw), (jj*Nw5):(jj*Nw5+Nw)].flatten()
                counter += 1

    dataWhite,junk = whitener.raw2normalized(stackedSmallSamples.T)
    # dataWhite.shape = (Nw^2, 4*Nsamples). Now with one example per COLUMN!

    # Represent (pooled is one example per COLUMN)
    hidden,pooled = tica.getRepresentation(dataWhite)

    # Create the input data for the next layer.

    # 1. Subsample (via bed of nails, no averaging) each of the four
    # overlapping windows by a factor of two in each direction,
    # creating the same number of total dimensions (exact for even
    # dimensions, approximate for odd).  Example: pooled size is
    # 15x15, downsample to 8x8, stack with four neighbors for a total
    # of 8x8x4 = 256 (instead of 225) dimensions.
    print 'TODO: LCN'
    hiddenSize = myres['params']['hiddenISize']
    if pooled.shape[0] != hiddenSize**2:
        raise Exception('Expected pooled layer to be %dx%d, but it is length %d'
                        % (hiddenSize, hiddenSize, pooled.shape[0]))
    downsampleSize = (hiddenSize+1)/2    # round up
    downsampleSizeSq = downsampleSize**2
    downsampled = zeros((downsampleSize**2, Nsamples*4))   # store in columns for use with neighbor matrix
    
    gaussNeighbors = neighborMatrix((downsampleSize,downsampleSize), 2.0, gaussian=True)
    for smallIdx in xrange(4*Nsamples):
        poolEmbedded = reshape(pooled[:,smallIdx], (hiddenSize, hiddenSize))     # 15x15
        downsampled[:,smallIdx] = poolEmbedded[::2,::2].flatten()                # 8x8 -> 64 column

    # 2. LCN
    vv = downsampled - dot(gaussNeighbors, downsampled)
    sig = sqrt(dot(gaussNeighbors, vv**2))
    cc = .01     # ss = sorted(sig.flatten()); ss[len(ss)/10] = 0.026 in one test. So .01 seems about right.
    yy = vv / maximum(cc, sig)
    
    # 3. Stack together 4x (one example per COLUMN)
    nextLayerInput = zeros((downsampleSizeSq * 4, Nsamples))
    for smallIdx in xrange(4*Nsamples):
        col    = smallIdx / 4
        offset = smallIdx % 4
        nextLayerInput[(offset*downsampleSizeSq):((offset+1)*downsampleSizeSq),col] = yy[:,smallIdx]





    print 'time: %f seconds' % (time.time()-tic)
    
    print 'HERE!'
    print 'Next: sample four windows, whiten, push through tica, pool, and subsample.'
    
    #pdb.set_trace()

    #resman.stop()
    

if __name__ == '__main__':
    main()
