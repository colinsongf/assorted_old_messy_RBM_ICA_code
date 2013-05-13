#! /usr/bin/env python

import sys
import imp
import pdb
import argparse
from numpy import zeros, prod, reshape

from GitResultsManager import resman
from util.misc import dictPrettyPrint, importFromFile, relhack
from util.dataPrep import PCAWhiteningDataNormalizer
from util.dataLoaders import loadFromPklGz, saveToFile
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from stackedLayers import StackedLayers



def main():
    parser = argparse.ArgumentParser(description='Visualizes a previously trained StackedLayers model.')
    parser.add_argument('--name', type = str, default = 'junk',
                        help = 'Name for GitResultsManager results directory (default: junk)')
    parser.add_argument('load', type = str,
                        help = 'Which previously saved StackedLayers object to load')

    args = parser.parse_args()


    resman.start(args.name)

    print 'Loading StackedLayers object from %s' % args.load
    sl = loadFromPklGz(args.load)

    print 'Loaded the following StackedLayers:'
    sl.printStatus()

    print 'VIS HERE!'
    
    resman.stop()



if __name__ == '__main__':
    main()
