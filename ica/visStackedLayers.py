#! /usr/bin/env python

import sys
import os
import imp
import pdb
import argparse
import shutil
from IPython import embed
from numpy import *

from GitResultsManager import resman
from util.misc import dictPrettyPrint, importFromFile, relhack
from util.dataPrep import PCAWhiteningDataNormalizer
from util.dataLoaders import loadFromPklGz, saveToFile
from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
from stackedLayers import StackedLayers



def main():
    parser = argparse.ArgumentParser(description='Visualizes a trained StackedLayers model.')
    parser.add_argument('--name', type = str, default = 'junk',
                        help = 'Name for GitResultsManager results directory (default: junk)')
    #parser.add_argument('--quick', action='store_true', help = 'Enable quick mode (default: off)')
    parser.add_argument('--nodiary', action='store_true', help = 'Disable diary (default: diary is on)')
    parser.add_argument('stackedLayersFilename', type = str,
                        help = 'File where a StackedLayers model was stored, something like stackedLayers.pkl.gz')
    parser.add_argument('command', type = str, default = 'embed', choices = ['visall', 'embed'], nargs='?',
                        help = 'What to do: one of {visall (save all plots), embed (drop into shell)}. Default: embed.')

    args = parser.parse_args()

    resman.start(args.name, diary = not args.nodiary)
    saveDir = resman.rundir

    print 'Loading StackedLayers from %s' % args.stackedLayersFilename
    sl = loadFromPklGz(args.stackedLayersFilename)

    print 'Loaded these StackedLayers:'
    sl.printStatus()

    if args.command == 'embed':
        embed()
    elif args.command == 'visall':
        sl.visAll(saveDir)
    else:
        print 'Unknown command:', args.command

    resman.stop()



if __name__ == '__main__':
    main()
