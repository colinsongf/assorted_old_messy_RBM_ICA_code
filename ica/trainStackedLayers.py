#! /usr/bin/env ipythonpl

import sys
import os
import imp
import pdb
import argparse
import shutil
from numpy import zeros, prod, reshape

from GitResultsManager import resman
from util.misc import dictPrettyPrint, importFromFile, relhack
from util.dataPrep import PCAWhiteningDataNormalizer
from util.dataLoaders import loadFromPklGz, saveToFile
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from stackedLayers import StackedLayers



def main():
    parser = argparse.ArgumentParser(description='Trains a StackedLayers model.')
    parser.add_argument('layerFilename', type = str,
                        help = 'File defining layers, something like tica-10-15.layers')
    parser.add_argument('trainParamsFilename', type = str,
                        help = 'File defining training parameters, something like tica-10-15.trainparams')
    parser.add_argument('--name', type = str, default = 'junk',
                        help = 'Name for GitResultsManager results directory (default: junk)')
    parser.add_argument('--load', type = str, default = '',
                        help = ('Load a previously created StackedLayers object. This can ' +
                                'be used to resume training a previously checkpointed, ' +
                                'partially trained StackedLayers object (default: none)'))
    parser.add_argument('--quick', action='store_true', help = 'Enable quick mode (default: off)')
    parser.add_argument('--nodiary', action='store_true', help = 'Disable diary (default: diary is on)')

    args = parser.parse_args()

    resman.start(args.name, diary = not args.nodiary)
    saveDir = resman.rundir


    layerDefinitions = importFromFile(args.layerFilename, 'layers')
    trainParams = importFromFile(args.trainParamsFilename, 'trainParams')

    shutil.copyfile(args.layerFilename,       os.path.join(saveDir, 'params.layers'))
    shutil.copyfile(args.trainParamsFilename, os.path.join(saveDir, 'params.trainparams'))

    if args.load:
        print 'Loading StackedLayers object from %s' % args.load
        sl = loadFromPklGz(args.load)
    else:
        print 'Creating new StackedLayers object'
        sl = StackedLayers(layerDefinitions)

    sl.printStatus()

    sl.train(trainParams, saveDir = saveDir, quick = args.quick)

    resman.stop()




if __name__ == '__main__':
    main()
