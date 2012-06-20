#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

from ica import testIca
from rbm.utils import load_mnist_data
from rbm.ResultsManager import resman



if __name__ == '__main__':
    '''Demonstrate ICA on the MNIST data set.'''

    resman.start('junk', diary = True)
    datasets = load_mnist_data('../data/mnist.pkl.gz', shared = False)

    testIca(datasets = datasets,
            savedir = resman.rundir,     # comment out to show plots instead of saving
            smallImgHack = False,
            quickHack = False,
            )
    resman.stop()
