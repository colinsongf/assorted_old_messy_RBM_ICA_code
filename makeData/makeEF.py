#! /usr/bin/env python
#! /usr/local/bin/ipython --gui=wx

import os, pdb, gzip, sys
import re
from numpy import *
import cPickle as pickle

from util.dataLoaders import saveToFile



def randomEFSampleMatrix(dataPath, seed = 0, Nsamples = 10):
    random.seed(seed)

    xmlFiles = []
    pattern = re.compile('_[0-9]{5}\.xml$')
    for root, dirs, files in os.walk(dataPath):
        for fil in files:
            filename = os.path.join(root, fil)
            #print filename,
            if pattern.search(filename):
                #print 'yes'
                xmlFiles.append(filename)
            else:
                pass
                #print 'no'
    random.shuffle(xmlFiles)
    numXmlFiles = len(xmlFiles)
    print 'found', numXmlFiles, 'xml files, selecting random sample of', Nsamples
    if Nsamples > numXmlFiles:
        raise Exception('Did not find enough xml files at path %s' % dataPath)

    for ii, filename in enumerate(xmlFiles[:Nsamples]):
        print ii, filename
        pdb.set_trace()





    
    retX = zeros((Nsamples, Nw * Nw * Nw), dtype = float32)
    retY = zeros((Nsamples, 6), dtype = float32)

    # compute once to save time
    indexX, indexY, indexZ = mgrid[0:Nw,0:Nw,0:Nw]
    
    for ii in xrange(Nsamples):
        # randomly pick coordinates of cube
        tries = 0
        bounds = random.randint(0, Nw, (2,3))
        while not all(bounds[0,:] - bounds[1,:] >= minSize):
            bounds = random.randint(0, Nw, (2,3))
            if tries > 100:
                raise Exception('failed too many times')

        # sort bounds in (min, max) order
        bounds.sort(0)

        thisImg = paintCube(indexX, indexY, indexZ, bounds)

        retX[ii,:] = thisImg.flatten()
        retY[ii,:] = bounds.flatten()
    
    return retX, retY



def main(dataPath):
    saveToFile('../data/endlessforms/train_real_50.pkl.gz',    randomEFSampleMatrix(dataPath = dataPath, Nsamples = 50))
    saveToFile('../data/endlessforms/test_real_50.pkl.gz',     randomEFSampleMatrix(dataPath = dataPath, Nsamples = 50))



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Usage:\n    %s path_to_directory_of_shapes    # create and save EF data.' % (sys.argv[0])
        sys.exit(1)

    dataPath = sys.argv[1]

    main(dataPath)

