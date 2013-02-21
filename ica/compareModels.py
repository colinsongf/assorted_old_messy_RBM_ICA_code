#! /usr/bin/env ipythonpl

import pdb
import os
import time
from numpy import *

from GitResultsManager import resman

from util.dataLoaders import loadCifarData, loadCifarDataMonochrome, loadCifarDataSubsets



def main():
    data = loadFromPklGz('../data/upson_rovio_2/train_10_50000_1c.pkl.gz')
    data = data.T   # Make into one example per column
    data = data[:,:5000]      # HACK!!!!!!!!!
    
    hiddenISize = 8
    hiddenJSize = 8
    lambd = .05
    neighborhoodSize = 1.5
    print '\nChosen TICA parameters'
    
    #hiddenISize = random.randint(4, 25+1)
    #hiddenJSize = random.randint(10, 30+1)
    #lambd = .1 * 2 ** random.randint(-5, 5+1)
    #neighborhoodSize = random.uniform(.1,3)
    #print '\nRandomly selected TICA parameters'

    for key in ['hiddenISize', 'hiddenJSize', 'lambd', 'neighborhoodSize']:
        print '  %20s: %s' % (key, locals()[key])
    
    random.seed(0)
    tica = TICA(imgShape = (15, 15, 3),
                hiddenLayerShape = (hiddenISize, hiddenJSize),
                neighborhoodParams = ('gaussian', neighborhoodSize, 0, 0),
                lambd = lambd,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    tica.run(data, plotEvery = 50, maxFun = 300, normData = True, whiten = True)






    datasets, classNames = loadCifarDataSubsets('../data/cifar-10-batches-py/',
                                                (16,16),
                                                ((0,0), (0,16), (16,0), (16,16)))

    pdb.set_trace()



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
