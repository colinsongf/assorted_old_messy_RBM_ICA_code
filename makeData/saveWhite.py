#! /usr/bin/env python

import ipdb as pdb
import sys
from numpy import *

from util.cache import cached
from util.dataLoaders import loadUpsonData, loadAtariData, saveToFile
from util.dataPrep import printDataStats, PCAWhiteningDataNormalizer



def main():
    ''''''

    try:
        dataLoaderName, dataPath, savePathWhite, savePathWhiteNormed, savePathWhiter = sys.argv[1:6]
    except:
        print 'usage:         dataLoaderName, dataPath, savePathWhite, savePathWhiteNormed, savePathWhiter'
        sys.exit(1)

    dataLoader = globals()[dataLoaderName]   # convert string to function
    data = dataLoader(dataPath)
    #data = data[:,:1000]; print 'HACKK!'

    print 'Raw data stats:'
    printDataStats(data)
    
    # Whiten with PCA
    whiteningStage = PCAWhiteningDataNormalizer(data)
    saveToFile(savePathWhiter, whiteningStage)

    # Non-normed
    dataWhite, junk = whiteningStage.raw2normalized(data, unitNorm = False)
    print 'White data stats:'
    printDataStats(dataWhite)
    saveToFile(savePathWhite, dataWhite)
    del dataWhite
    
    # Normed
    dataWhiteNormed, junk = whiteningStage.raw2normalized(data, unitNorm = True)
    print 'White normed data stats:'
    printDataStats(dataWhiteNormed)
    saveToFile(savePathWhiteNormed, dataWhiteNormed)
    del dataWhiteNormed

    print 'done.'



if __name__ == '__main__':
    main()
