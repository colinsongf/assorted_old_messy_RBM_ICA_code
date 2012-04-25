#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

import sys, os
import Image, pdb
from numpy import *

from utils import tile_raster_images, imagesc, load_mnist_data
from upsonRbm import loadUpsonData
from rbm import test_rbm
from ResultsManager import resman
from pca import PCA

from matplotlib import pyplot



def main():
    if len(sys.argv) <= 4:
        print 'Usage: %s pcaDims epsilon n_hidden learningRate' % sys.argv[0]
        sys.exit(1)
    

    # loads data like datasets = ((train_x, train_y), ([], None), (test_x, None))
    datasets = loadUpsonData('../data/upson_rovio_1/train_15_50000.pkl.gz',
                             '../data/upson_rovio_1/test_15_50000.pkl.gz')
    img_dim = 15   # must match actual size of training data

    print 'done loading.'

    pcaDims = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    pca = PCA(datasets[0][0])  # train
    datasets[0][0] = pca.toZca(datasets[0][0], pcaDims, epsilon = epsilon) # train
    datasets[1][0] = pca.toZca(datasets[1][0], pcaDims, epsilon = epsilon) if len(datasets[1][0]) > 0 else array([]) # valid
    datasets[2][0] = pca.toZca(datasets[2][0], pcaDims, epsilon = epsilon) # test

    # plot mean and principle components
    image = Image.fromarray(tile_raster_images(
             X = pca.meanAndPc(pcaDims).T,
             img_shape = (img_dim,img_dim),tile_shape = (10,10),
             tile_spacing=(1,1)))
    image.save(os.path.join(resman.rundir, 'meanAndPc.png'))
    
    # plot fractional stddev in PCA dimensions
    pyplot.semilogy(pca.fracStd, 'bo-')
    if pcaDims is not None:
        pyplot.axvline(pcaDims)
    pyplot.savefig(os.path.join(resman.rundir, 'fracStd.png'))
    pyplot.clf()
    
    
    test_rbm(datasets = datasets,
             training_epochs = 45,
             img_dim = img_dim,
             n_hidden = int(sys.argv[3]),
             learning_rate = float(sys.argv[4]),
             output_dir = resman.rundir,
             quickHack = False,
             visibleModel = 'real',
             initWfactor = .01,
             imgPlotFunction = lambda xx: pca.fromZca(xx, numDims = pcaDims, epsilon = epsilon))



if __name__ == '__main__':
    resman.start('junk', diary = True)
    main()
    resman.stop()

