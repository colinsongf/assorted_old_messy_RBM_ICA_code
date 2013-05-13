#! /usr/bin/env python

import sys
import imp
import pdb
import argparse
from numpy import zeros, prod, reshape

from GitResultsManager import resman
from util.misc import dictPrettyPrint, importFromFile, relhack
from util.dataPrep import PCAWhiteningDataNormalizer
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from stackedLayers import StackedLayers



def main(layerFilename, trainParamsFilename, quick = False):
    if quick:
        print 'WARNING: running in --quick mode, chopping all training data to 1000 examples!!'
    layers = importFromFile(layerFilename, 'layers')
    trainParams = importFromFile(trainParamsFilename, 'trainParams')

    sl = StackedLayers(layers)

    sl.printStatus()

    sl.train(trainParams, quick = quick)

    pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='layer sandbox...')
    parser.add_argument('layerFilename', type = str,
                        help='file defining layers, something like tica-10-15.layers')
    parser.add_argument('trainParamsFilename', type = str,
                        help='file defining training parameters, something like tica-10-15.trainparams')
    parser.add_argument('--quick', action='store_true')

    args = parser.parse_args()

    #resman.start('junk')
    
    main(args.layerFilename, args.trainParamsFilename, args.quick)

    #resman.stop()
    
