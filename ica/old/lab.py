#! /usr/bin/env python

import pdb
import os
import gc
from numpy import zeros, prod, reshape
from IPython import embed

from util.dataLoaders import loadFromPklGz, saveToFile
from util.misc import dictPrettyPrint, importFromFile, relhack
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, NormalizingLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from GitResultsManager import resman
from util.dataPrep import PCAWhiteningDataNormalizer
from stackedLayers import StackedLayers



def main():
    dirs = [name for name in os.listdir('results') if os.path.isdir(os.path.join('results', name))]
    print 'last few results:'
    for dir in sorted(dirs)[-10:]:
        print '  ' + dir

    resman.start('junk', diary = False)
    saveDir = resman.rundir
    
    embed()
    
    resman.stop()


if __name__ == '__main__':
    main()
