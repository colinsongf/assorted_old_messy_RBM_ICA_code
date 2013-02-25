#! /usr/bin/env python

import os
import pdb
from numpy import *
import cProfile
import pstats
import tempfile

#from matplotlib import pyplot

from util.cache import memoize, cached
from pca import PCA



#@profile
def testPca():
    random.seed(0)
    
    NN = 5000
    dim = 200
    
    data = random.randn(NN,dim)

    pca = cached(PCA, data)
    #pca = PCA(data)

    pc = pca.pc()

    asPc = pca.toPC(data)

    dataWhite = pca.toZca(data, epsilon = 1e-6)



if __name__ == '__main__':
    justRun = False

    if justRun:
        testPca()
    else:
        #cProfile.run('testPca', 'profile.log')
        fd,tmpFile = tempfile.mkstemp()
        prof = cProfile.run('testPca()', tmpFile)

        prof = pstats.Stats(tmpFile)

        prof.sort_stats('cumulative').print_stats(10)

        prof.sort_stats('cumulative').print_callers(10)

        os.unlink(tmpFile)
