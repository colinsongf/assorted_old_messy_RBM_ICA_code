#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys
from numpy import *
from matplotlib import pyplot
from PIL import Image

from sklearn.decomposition import FastICA

from rbm.ResultsManager import resman
from rbm.pca import PCA
from rbm.utils import tile_raster_images, load_mnist_data, saveToFile



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
    
    image = Image.fromarray(tile_raster_images(
             X = train_set_x,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'data_raw.png'))
    image.show()

    pyplot.figure()
    for ii in range(20):
        idx = random.randint(0, nDim)
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
        idx = random.randint(0, nDim)
        pyplot.subplot(4,5,ii+1)
        pyplot.title('white dim %d' % idx)
        pyplot.hist(xWhite[:,idx])
    if savedir: pyplot.savefig(os.path.join(savedir, 'data_white_hist.png'))

    image = Image.fromarray(tile_raster_images(
             X = xWhite,
             img_shape = (imgDim,imgDim), tile_shape = (10,15),
             tile_spacing=(1,1)))
    if savedir:  image.save(os.path.join(savedir, 'data_white.png'))
    image.show()


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
        idx = random.randint(0, nDim)
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



if __name__ == '__main__':
    '''Demonstrate ICA on the MNIST data set.'''

    resman.start('junk', diary = False)
    datasets = load_mnist_data('../data/mnist.pkl.gz', shared = False)

    main(datasets = datasets,
         savedir = resman.rundir,     # comment out to show plots instead of saving
         smallImgHack = False,
         quickHack = False,
         )
    resman.stop()
