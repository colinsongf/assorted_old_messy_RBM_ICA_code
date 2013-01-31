#! /usr/bin/env ipythonpl

import pdb
import os
import time
from numpy import *

from GitResultsManager import resman

from util.dataLoaders import loadCifarData, loadCifarDataMonochrome, loadCifarDataSubsets



def main():
    datasets, classNames = loadCifarDataSubsets('../data/cifar-10-batches-py/',
                                                (16,16),
                                                ((0,0), (0,16), (16,0), (16,16)))

    pdb.set_trace()



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
