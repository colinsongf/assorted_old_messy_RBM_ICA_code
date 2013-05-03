#! /usr/bin/env python

import sys
import imp
import ipdb as pdb
import argparse

#from utils import loadFromFile
#from squaresRbm import loadPickledData
from GitResultsManager import resman
#from util.plotting import plot3DShapeFromFlattened
from util.dataPrep import PCAWhiteningDataNormalizer  #, printDataStats
from layers import Layer, DataLayer, UpsonData3, WhiteningLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer



layerClassNames = {'data':       'dataClass',       # load the class specified by DataClass
                   'whitener':   'whitenerClass',   # load the class specified by whitenerClass 
                   'tica':       TicaLayer,
                   'downsample': DownsampleLayer,
                   'lcn':        LcnLayer,
                   'concat':     ConcatenationLayer,
                   }



class StackedLayers(object):

    def __init__(self, layerList):

        self.layers = []

        # 0. Check for duplicate names
        layerNames = [layer['name'] for layer in layerList]
        dups = set([x for x in layerNames if layerNames.count(x) > 1])
        if len(dups) > 0:
            raise Exception('Duplicate layer names: %s' % dups)

        # 1. Construct all layers
        for ii, layerDict in enumerate(layerList):
            print 'constructing layer %d:' % ii
            print '  ', layerDict

            layerClass = layerClassNames[layerDict['type']]
            if isinstance(layerClass, basestring):
                actualLayerClass = eval(layerDict[layerClass])
                layer = actualLayerClass(layerDict)
            else:
                layer = layerClass(layerDict)

            # Checks
            if ii == 0 and not layer.isDataLayer:
                raise Exception('First layer must be data layer, but it is %s' % repr(layer))
            if ii != 0 and layer.isDataLayer:
                raise Exception('Only the first layer can be a data layer, but layer %d is %s' % (ii, repr(layer)))

            # Populate inputSize, outputSize, and seesPixels
            if ii == 0:
                # For data layer
                layer.outputSize = layer.getOutputSize()
            else:
                prevLayer = self.layers[ii-1]

                # only for non-data layers
                layer.inputSize = prevLayer.outputSize
                layer.outputSize = layer.calculateOutputSize(layer.inputSize)
                prevLayerSees = prevLayer.outputSize if prevLayer.isDataLayer else prevLayer.seesPixels
                layer.seesPixels = layer.calculateSeesPixels(prevLayerSees)

            self.layers.append(layer)

        # 2. Post construction checks...
        for ii, layer in enumerate(self.layers):
            pass
        print 'layers look good (purely cursory check)'


    def printStatus(self):
        for ii, layer in reversed(list(enumerate(self.layers))):
            print 'layer %d: %-20s' % (ii, '%s (%s)' % (layer.name, layer.layerType)),
            if ii == 0:
                st = 'outputs size %s' % repr(layer.outputSize),
            else:
                st = 'maps size %s -> %s' % (repr(layer.inputSize), repr(layer.outputSize)),
            print '%-32s' % st,
            if layer.trainable:
                print 'trainable'
            else:
                print 'not trainable'
                



def importFromFile(filename, objectName):
    try:
        with open(filename, 'r') as ff:
            fileText = ff.read()
    except IOError:
        print 'Could not open file "%s". Are you sure it exists?' % filename
        raise

    try:
        exec(compile(fileText, 'contents of file: %s' % filename, 'exec'))
    except:
        print 'Tried to execute file "%s" but got this error:' % filename
        raise
        
    if not objectName in locals():
        raise Exception('file "%s" did not define the %s variable' % (layerFilename, objectName))

    return locals()[objectName]



def main(layerFilename, trainParamFilename):
    layers = importFromFile(layerFilename, 'layers')
    trainParam = importFromFile(trainParamFilename, 'trainParam')

    sl = StackedLayers(layers)

    sl.printStatus()

    #sl.train(trainParam)

    pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='layer sandbox...')
    parser.add_argument('layerFilename', type = str,
                        help='file defining layers, something like tica-10-15.layers')
    parser.add_argument('trainParamFilename', type = str,
                        help='file defining training parameters, something like tica-10-15.trainparam')
    args = parser.parse_args()

    #resman.start('junk')
    
    main(args.layerFilename, args.trainParamFilename)

    #resman.stop()
    
