#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb

from ica import testIca
from rbm.utils import loadUpsonData
from util.dataLoaders import loadCifarData, loadCifarDataMonochrome
from GitResultsManager import resman



if __name__ == '__main__':
    '''Demonstrate ICA on the Upson data set.'''

    resman.start('junk', diary = False)

    datasets, classNames = loadCifarDataMonochrome('../data/cifar-10-batches-py/')

    testIca(datasets = datasets,
            savedir = resman.rundir,     # comment out to show plots instead of saving
            smallImgHack = False,
            quickHack = False,
            )
    resman.stop()
