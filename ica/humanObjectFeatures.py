#! /usr/bin/env python

import sys
import os
import ipdb as pdb
import argparse
import shutil
from IPython import embed
from numpy import *
from PIL import Image

from GitResultsManager import resman

from layers import DataArrangement
#from util.misc import dictPrettyPrint, importFromFile, relhack
#from util.dataPrep import PCAWhiteningDataNormalizer
from util.dataLoaders import loadFromPklGz, saveToFile, approxNormalizeCS294
#from layers import layerClassNames, DataArrangement, Layer, DataLayer, UpsonData3, NormalizingLayer, PCAWhiteningLayer, TicaLayer, DownsampleLayer, LcnLayer, ConcatenationLayer
#from stackedLayers import StackedLayers
#from visualize import plotImageData, plotTopActivations, plotGrayActivations, plotReshapedActivations, plotActHist



def makeFeats(dataDir, featsFilename, outputFilename, saveDir = None, quick = False, stackedLayers = None,
              featsBase = True, featsConst = 0, featsRand = 0, featsSL = False):
    nLines = 0
    nMismatchedIds = 0
    output = []
    with open(featsFilename, 'r') as ff:
        lines = ff.readlines()

    for ii, line in enumerate(lines):
        if quick and ii > 1:
            break
        line = line.strip()
        #print line

        vals = line.split(',')
        intVals = [int(round(float(st))) for st in vals]
        linePrefix = vals[:5]
        activityId, segId, objId, frameNum, objIdDup, baseFeatures = vals[0], intVals[1], intVals[2], intVals[3], intVals[4], vals[5:]
        assert len(baseFeatures) == 18
        #assert objId == objIdDup
        if objId != objIdDup:
            nMismatchedIds += 1
            #print line.strip()
        objBoundingBox = [int(st) for st in baseFeatures[:4]]
        imPath = os.path.join(dataDir, 'by_id', activityId, 'RGB_%d.png' % frameNum)

        feats = []
        
        # Get all chosen features:
        if featsBase:
            feats.extend(baseFeatures)
        if featsConst:
            feats.extend(featConst(featsConst))
        if featsRand:
            feats.extend(featRand(featsRand))
        if featsSL:
            feats.extend(featSLRep(imPath, objId, objBoundingBox, stackedLayers))
            
        output.append((linePrefix, feats))
        #print ii, '%s + %s' % (linePrefix, feats)

        nLines += 1
        if ii == 0:
            #embed()
            pass
        if ii % 100 == 0:
            print 'Processed %4d/%d lines' % (ii, len(lines))

    print 'Read %d lines (%d mismatched obj IDs)' % (nLines, nMismatchedIds)

    nfeats = None
    with open(outputFilename, 'w') as ff:
        for linePrefix, feats in output:
            if nfeats is None:
                nfeats = len(feats)
            else:
                assert nfeats == len(feats), 'Non-uniform number of features.'
            newFeatureStrList = []
            for feat in feats:
                if isinstance(feat, float):
                    newFeatureStrList.append('%f' % feat)
                elif isinstance(feat, int):
                    newFeatureStrList.append('%d' % feat)
                elif isinstance(feat, basestring):
                    newFeatureStrList.append('%s' % feat)
                else:
                    raise Exception('Unknown feature type: %s' % type(feat))
            ff.write('%s\n' % ','.join(linePrefix + newFeatureStrList))

    print 'Wrote %d lines, %d feats per line, to %s' % (len(output), nfeats, outputFilename)

    if saveDir:
        shutil.copy(outputFilename, saveDir)



def featConst(number = 5):
    return [ii for ii in range(number)]



def featRand(number = 5):
    return [random.uniform(0, 1) for ii in range(number)]



#shown = []
def featAvgIntensity(imPath, objId, objBoundingBox):
    '''Very simple feature: just returns the average intensity of the patch.'''

    if '0505003751' in imPath or '0510181539' in imPath:
        # Missing images...
        return [0]

    im = Image.open(imPath)
    objCrop = im.crop(objBoundingBox)
        
    #if '0510181236' in imPath and ('RGB_350' in imPath or 'RGB_254' in imPath):
    #    print 'showing', imPath, objId, objBoundingBox
    #    if imPath not in shown:
    #        im.show()
    #        shown.append(imPath)
    #    objCrop.show()
    #    #pdb.set_trace()
    left,upper,right,lower = objBoundingBox
    if left == right or upper == lower:
        #raise Exception('null bounding box')
        return [0]

    arr = array(objCrop)

    return [arr.mean()]



def featSLRep(imPath, objId, objBoundingBox, stackedLayers):
    '''Returns the representation of the patch using the provided StackedLayers object'''

    if not stackedLayers:
        raise Exception('Need a StackedLayers object, but None was provided! (Make sure you load one using the relevant command line option).')

    slInputSize = stackedLayers.layers[1].inputSize
    slOutputSize = prod(stackedLayers.layers[-1].outputSize)

    if '0505003751' in imPath or '0510181539' in imPath:
        # Missing images...
        return [0] * slOutputSize

    im = Image.open(imPath)
    objCrop = im.crop(objBoundingBox)
    left,upper,right,lower = objBoundingBox

    # Check for null bounding box (e.g. when object is occluded)
    if left == right or upper == lower:
        return [0] * slOutputSize

    objCrop.thumbnail(slInputSize, Image.ANTIALIAS)
    objCropGray = objCrop.convert('L')
    arr = array(objCropGray, dtype='float')

    paddingAmounts = [goal-cur for goal,cur in zip(slInputSize, arr.shape)]
    paddingWidths = [(pa/2, int(round((pa+.5)/2))) for pa in paddingAmounts]   # To approximately center in each direction

    # Pad with mean to appropriate size
    arrpad = pad(arr, paddingWidths, mode='constant', constant_values=(arr.mean(),))
    arrnrm = approxNormalizeCS294(arrpad, 255)
    
    rep, dataArrangement = stackedLayers.forwardProp(reshape(arrnrm.flatten(), (-1,1)), DataArrangement((1,1), 1), quiet = True)

    #embed()

    return rep.flatten().tolist()



def main():
    parser = argparse.ArgumentParser(description='Makes features for the human activity deteciton dataset. Example usage:\n./humanObjectFeatures.py /path/to/data_obj_feats.txt')
    parser.add_argument('--name', type = str, default = 'junk',
                        help = 'Name for GitResultsManager results directory (default: junk)')
    parser.add_argument('--quick', action='store_true', help = 'Enable quick mode (default: off)')
    parser.add_argument('--nodiary', action='store_true', help = 'Disable diary (default: diary is on)')

    #parser.add_argument('stackedLayersFilename', type = str,
    #                    help = 'File where a StackedLayers model was stored, something like stackedLayers.pkl.gz')
    #parser.add_argument('command', type = str, default = 'embed', choices = ['visall', 'embed'], nargs='?',
    #                    help = 'What to do: one of {visall (save all plots), embed (drop into shell)}. Default: embed.')

    parser.add_argument('--stackedlayers', type = str, default = '',
                        help = 'Path to a *.pkl.gz file containing a pickeled StackedLayers object to load (default: None)')
    parser.add_argument('--outfile', type = str, default = 'data_objs_feats_plus.txt',
                        help = 'What to name the output file (default: data_objs_feats_plus.txt)')
    parser.add_argument('dataDir', type = str, default = 'data',
                        help = 'Where to look for the "by_id" directory')
    parser.add_argument('data_obj_feats_file', type = str,
                        help = 'Which data_objs_feats.txt file to load')

    args = parser.parse_args()

    resman.start(args.name, diary = not args.nodiary)
    saveDir = resman.rundir

    #print 'Loading StackedLayers from %s' % args.stackedLayersFilename
    #sl = loadFromPklGz(args.stackedLayersFilename)
    #print 'Loaded these StackedLayers:'
    #sl.printStatus()

    stackedLayers = None
    if args.stackedlayers:
        stackedLayers = loadFromPklGz(args.stackedlayers)

    makeFeats(dataDir = args.dataDir, featsFilename = args.data_obj_feats_file,
              saveDir = saveDir, outputFilename = args.outfile, quick = args.quick,
              stackedLayers = stackedLayers,
              featsBase = True, featsConst = 0, featsRand = 0, featsSL = False)
    
    resman.stop()



if __name__ == '__main__':
    main()
