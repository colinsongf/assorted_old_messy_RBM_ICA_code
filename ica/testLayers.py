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



class StackedLayers(object):

    def __init__(self, layerList):

        self.layers = []

        # Check for duplicate names
        layerNames = [layer['name'] for layer in layerList]
        dups = set([x for x in layerNames if layerNames.count(x) > 1])
        if len(dups) > 0:
            raise Exception('Duplicate layer names: %s' % dups)
        
        for ii, layerDict in enumerate(layerList):
            print 'layer %d:' % ii
            print '  ', layerDict

            layerClass = layerClassNames[layerDict['type']]
            if isinstance(layerClass, basestring):
                actualLayerClass = eval(layerDict[layerClass])
                thisLayer = actualLayerClass(layerDict)
            else:
                thisLayer = layerClass(layerDict)

            self.layers.append(thisLayer)

        # Check layer compatability
        for ii, layer in enumerate(self.layers):
            if ii == 0 and not layer.isDataLayer:
                raise Exception('First layer must be data layer, but it is %s' % repr(layer))
            if ii != 0 and layer.isDataLayer:
                raise Exception('Only the first layer can be a data layer, but layer %d is %s' % (ii, repr(layer)))

        print 'layers look good (purely cursory check)'






class Layer(object):

    trainable = False   # override if desired
    isDataLayer = False
    
    def __init__(self):
        pass



######################
# Data
######################

class DataLayer(Layer):

    isDataLayer = True

    def __init__(self):
        super(DataLayer, self).__init__()
        pass


class UpsonData3(DataLayer):

    def __init__(self, params):
        self.colors = params['colors']



######################
# Whitening
######################

class WhiteningLayer(Layer):

    trainable = True

    def __init__(self, params):
        pass


class PCAWhiteningLayer(WhiteningLayer):

    def __init__(self, params):
        pass



######################
# Learning
######################

class TicaLayer(Layer):

    trainable = True

    def __init__(self, params):
        pass



######################
# Downsampling, LCN, Concatenation
######################

class DownsampleLayer(Layer):

    def __init__(self, params):
        pass



class LcnLayer(Layer):

    def __init__(self, params):
        pass



class ConcatenationLayer(Layer):

    def __init__(self, params):
        pass



layerClassNames = {'data':       'dataClass',       # load the class specified by DataClass
                   'whitener':   'whitenerClass',   # load the class specified by whitenerClass 
                   'tica':       TicaLayer,
                   'downsample': DownsampleLayer,
                   'lcn':        LcnLayer,
                   'concat':     ConcatenationLayer,
                   }



def main(layerFilename):
    try:
        with open(layerFilename, 'r') as ff:
            layerFileText = ff.read()
    except IOError:
        print 'Could not open layer file "%s". Are you sure it exists?' % layerFilename
        sys.exit(1)


    try:
        exec(compile(layerFileText, 'contents of file: %s' % layerFilename, 'exec'))
    except:
        print 'Tried to execute layer definition file "%s" but got this error:' % layerFilename
        raise
        
    if not 'layers' in locals():
        print 'file "%s" did not define the layers variable' % layerFilename
        sys.exit(1)
    #layerDefModule = imp.load_source('layerDefModule', layerFilename)
    #layers = layerDefModule.layers

    #print 'got layers'
    #print layers

    stackedLayers = StackedLayers(layers)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='layer sandbox...')
    parser.add_argument('layerFilename', type = str,
                        help='file defining layers, something like tica-10-15.param')
    args = parser.parse_args()

    #resman.start('junk')
    
    main(args.layerFilename)

    #resman.stop()
    
