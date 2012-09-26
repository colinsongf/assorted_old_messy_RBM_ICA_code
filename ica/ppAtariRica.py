#! /usr/bin/env ipython

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys, time
from numpy import *
from PIL import Image
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
import pp

from rica import RICA
from util.ResultsManager import resman, fmtSeconds
from util.plotting import tile_raster_images
from util.dataLoaders import loadFromPklGz, saveToFile



def ricaRun(randSeed, resman):
    data = loadFromPklGz('../data/atari/mspacmantrain_15_50000_3c.pkl.gz')
    data = data.T   # Make into one example per column

    print 'randSeed is', randSeed
    random.seed(randSeed)
    print 'making dirs'
    thisdir = os.path.join(resman.rundir, 'seed-%03d' % randSeed)
    os.makedirs(thisdir)
    rica = RICA(imgShape = (15, 15, 3),
                nFeatures = 400,
                lambd = .05,
                epsilon = 1e-5,
                saveDir = thisdir)
    rica.run(data, maxFun = 3, whiten = True)

    return 'finished return value'


def testfn():
    print 'sys.path is', sys.path
    print 'cwd is', os.getcwd()


if __name__ == '__main__':
    resman.start('junk', diary = False)

    print 'local'
    testfn()

    ppservers = tuple(['xanthus-%d.mae.cornell.edu' % ii for ii in range(1,7)])
    job_server = pp.Server(ncpus=0, ppservers=ppservers)
    jobs = []    
    for ii in range(1):
        #jobs.append((ii,
        #             job_server.submit(ricaRun,
        #                               args = (ii, resman),
        #                               modules=('from rica import RICA',
        #                                        'from util.ResultsManager import resman, fmtSeconds',
        #                                        'from util.plotting import tile_raster_images',
        #                                        'from util.dataLoaders import loadFromPklGz, saveToFile',
        #                                        ),
        #                               ))
        #            )
        jobs.append((ii,
                     job_server.submit(testfn,
                                       ))
                    )
    print 'jobs is', jobs

    job_server.print_stats()

    for ii, job in jobs:
        print ii, job()
        
    job_server.print_stats()

    resman.stop()
