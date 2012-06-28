#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys
from numpy import *
from matplotlib import pyplot, mlab
from PIL import Image

from sklearn.decomposition import FastICA

from rbm.ResultsManager import resman
from rbm.pca import PCA
from rbm.utils import tile_raster_images, load_mnist_data, saveToFile, looser



def testIca(datasets, savedir = None, smallImgHack = False, quickHack = False):
    '''Test ICA on a given dataset.'''

    random.seed(1)

    # 0. Get data
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x,  test_set_y  = datasets[2]

    if quickHack:
        print '!!! Using quickHack !!!'
        train_set_x = train_set_x[:2500,:]
        if train_set_y is not None:
            train_set_y = train_set_y[:2500]
    if smallImgHack:
        print '!!! Using smallImgHack !!! (images will be misaligned)'
        train_set_x = train_set_x[:,:100]

    print ('(%d, %d, %d) %d dimensional examples in (train, valid, test)' % 
           (train_set_x.shape[0], valid_set_x.shape[0], test_set_x.shape[0], train_set_x.shape[1]))

    nDim = train_set_x.shape[1]
    imgDim = int(round(sqrt(nDim)))    # Might not always be true...

    randIdxRaw    = random.randint(0, nDim, 100)
    randIdxWhite  = random.randint(0, nDim, 100)
    randIdxSource = random.randint(0, nDim, 100)

    image = Image.fromarray(tile_raster_images(
             X = train_set_x,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'data_raw.png'))
    image.show()

    pyplot.figure()
    for ii in range(20):
        idx = randIdxRaw[ii]
        pyplot.subplot(4,5,ii+1)
        pyplot.title('raw dim %d' % idx)
        pyplot.hist(train_set_x[:,idx])
    if savedir: pyplot.savefig(os.path.join(savedir, 'data_raw_hist.png'))

    # 1. Whiten data
    print 'Whitening data with pca...'
    pca = PCA(train_set_x)
    xWhite = pca.toZca(train_set_x)
    print '  done.'

    pyplot.figure()
    for ii in range(20):
        idx = randIdxWhite[ii]
        pyplot.subplot(4,5,ii+1)
        pyplot.title('data white dim %d' % idx)
        pyplot.hist(xWhite[:,idx])
    if savedir: pyplot.savefig(os.path.join(savedir, 'data_white_hist.png'))

    image = Image.fromarray(tile_raster_images(
             X = xWhite,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'data_white.png'))
    image.show()

    # 1.1 plot hist
    pyplot.figure()
    pyplot.hold(True)
    pyplot.title('data white 20 random dims')
    histMax = 0
    histMin = 1e10
    for ii in range(20):
        idx = randIdxWhite[ii]
        hist, binEdges = histogram(xWhite[:,idx], bins = 20, density = True)
        histMax = max(histMax, max(hist))
        histMin = min(histMin, min(hist[hist != 0]))   # min non-zero entry
        binMiddles = binEdges[:-1] + (binEdges[1] - binEdges[0])/2
        #print ' %d from %f to %f' % (ii, min(binMiddles), max(binMiddles))
        pyplot.semilogy(binMiddles, hist, '.-')
    pyplot.axis('tight')
    ax = looser(pyplot.axis(), semilogy = True)
    xAbsMax = max(fabs(ax[0:2]))
    xx = linspace(-xAbsMax, xAbsMax, 100)
    pyplot.semilogy(xx, mlab.normpdf(xx, 0, 1), 'k', linewidth = 3)
    pyplot.axis((-xAbsMax, xAbsMax, ax[2], ax[3]))
    if savedir: pyplot.savefig(os.path.join(savedir, 'data_white_log_hist.png'))

    # 1.2 plot points
    pyplot.figure()
    pyplot.hold(True)
    pyplot.title('data white 20 random dims')
    nSamples = min(xWhite.shape[0], 1000)
    print 'data_white_log_points plotted with', nSamples, 'samples.'
    for ii in range(10):
        idx = randIdxWhite[ii]
        pyplot.plot(xWhite[:nSamples,idx],
                    ii + random.uniform(-.25, .25, nSamples), 'o')
    pyplot.axis('tight')
    if savedir: pyplot.savefig(os.path.join(savedir, 'data_white_log_points.png'))

    # 2. Fit ICA
    rng = random.RandomState(1)
    ica = FastICA(random_state = rng, whiten = False)
    print 'Fitting ICA...'
    ica.fit(xWhite)
    print '  done.'
    if savedir:  saveToFile(os.path.join(savedir, 'ica.pkl.gz'), ica)

    print 'Geting sources and mixing matrix...'
    sourcesWhite = ica.transform(xWhite)  # Estimate the sources
    #S_fica /= S_fica.std(axis=0)   # (should already be done)
    mixingMatrix = ica.get_mixing_matrix()
    print '  done.'

    sources = pca.fromZca(sourcesWhite)
    

    # 3. Show independent components and inferred sources
    image = Image.fromarray(tile_raster_images(
             X = mixingMatrix,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'ic_white.png'))
    image.show()
    image = Image.fromarray(tile_raster_images(
             X = mixingMatrix.T,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'ic_white.T.png'))
    image.show()
    image = Image.fromarray(tile_raster_images(
             X = pca.fromZca(mixingMatrix),
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'ic_raw.png'))
    image.show()
    image = Image.fromarray(tile_raster_images(
             X = pca.fromZca(mixingMatrix.T),
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'ic_raw.T.png'))
    image.show()

    pyplot.figure()
    for ii in range(20):
        idx = randIdxSource[ii]
        pyplot.subplot(4,5,ii+1)
        pyplot.title('sourceWhite %d' % idx)
        pyplot.hist(sourcesWhite[:,idx])
    if savedir: pyplot.savefig(os.path.join(savedir, 'sources_white_hist.png'))

    image = Image.fromarray(tile_raster_images(
             X = sourcesWhite,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'sources_white.png'))
    image.show()

    # 3.1 plot hist
    pyplot.figure()
    pyplot.hold(True)
    pyplot.title('sources white 20 random dims')
    histMax = 0
    histMin = 1e10
    for ii in range(20):
        idx = randIdxSource[ii]
        hist, binEdges = histogram(sourcesWhite[:,idx], bins = 20, density = True)
        histMax = max(histMax, max(hist))
        histMin = min(histMin, min(hist[hist != 0]))   # min non-zero entry
        binMiddles = binEdges[:-1] + (binEdges[1] - binEdges[0])/2
        #print ' %d from %f to %f' % (ii, min(binMiddles), max(binMiddles))
        pyplot.semilogy(binMiddles, hist, '.-')
    pyplot.axis('tight')
    ax = looser(pyplot.axis(), semilogy = True)
    xAbsMax = max(fabs(ax[0:2]))
    xx = linspace(-xAbsMax, xAbsMax, 100)
    pyplot.semilogy(xx, mlab.normpdf(xx, 0, 1), 'k', linewidth = 3)
    pyplot.axis((-xAbsMax, xAbsMax, ax[2], ax[3]))
    if savedir: pyplot.savefig(os.path.join(savedir, 'sources_white_log_hist.png'))

    # 3.2 plot points
    pyplot.figure()
    pyplot.hold(True)
    pyplot.title('sources white 20 random dims')
    nSamples = min(sourcesWhite.shape[0], 1000)
    print 'sources_white_log_points plotted with', nSamples, 'samples.'
    for ii in range(10):
        idx = randIdxWhite[ii]
        pyplot.plot(sourcesWhite[:nSamples,idx],
                    ii + random.uniform(-.25, .25, nSamples), 'o')
    pyplot.axis('tight')
    if savedir: pyplot.savefig(os.path.join(savedir, 'sources_white_log_points.png'))


    image = Image.fromarray(tile_raster_images(
             X = sources,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'sources_raw.png'))
    image.show()


    
    if savedir:
        print 'plots saved in', savedir
    else:
        import ipdb; ipdb.set_trace()
