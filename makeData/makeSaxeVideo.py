#! /usr/bin/env python

import os
import random
from numpy import array, zeros
from PIL import Image
import cPickle as pickle
import gzip
import ipdb as pdb



dataDir = '../data/saxe'
imsize = (640, 360)
chopPx = 10         # Amount to chop off edges (edges in this dataset are a bit weird)
dtype = 'uint8'
minBatchSize = 800
outputName = os.path.join(dataDir, 'batch_%02d.pkl.gz')

# Normalized output
normalizationOutputDtype = 'float16'
normalizationApproxMean = 117.0
normalizationMinMax = .8    # crop to, e.g.,-.8 to .8

dirNames = [
    'hummingbird',
    'illum_pan_tripod_flowers',
    'illum_tripod_leaves',
    'orbit_handheld_trunk',
    'orbit_turntable_checker',
    'orbit_turntable_leaves',
    'orbit_turntable_mouse',
    'orbit_turntable_mug1',
    'orbit_turntable_mug2',
    'pan_tripod_flowers',
    'pan_tripod_leaves',
    'rotate_handheld_flowers',
    'rotate_handheld_palm',
    'rotate_off_axis_tripod_flowers',
    'rotate_off_axis_tripod_leaves',
    'rotate_off_axis_tripod_palm',
    'slow_tilt_tripod_flowers',
    'squirrel',
    'still_handheld_flowers',
    'still_tripod_flowers',
    'still_tripod_leaves',
    'still_tripod_palm',
    'still_tripod_trunk',
    'tilt_tripod_flowers',
    'tilt_tripod_palm',
    'tilt_tripod_trunk',
    'translate_handheld_flowers',
    'translate_handheld_leaves',
    'translate_tripod_palm',
    'translate_tripod_trunk',
    'wind_motion_tripod_flowers',
    'zoom_tripod_flowers',
    'zoom_tripod_palm',
    'zoom_tripod_tree'
    ]


def loadImgs(dirname):
    arrayList = []
    for ii in xrange(10**7):
        try:
            im = Image.open(os.path.join(dirname, 'whitened%08d.png' % ii))
        except IOError:
            break
        assert im.size == imsize
        arrayList.append(array(im))
    numImages = len(arrayList)
    print 'loaded %d images from %s' % (numImages, dirname)
    if numImages == 0:
        raise Exception('Found no images.')
    frames = zeros((numImages, imsize[1]-chopPx*2, imsize[0]-chopPx*2),
                   dtype = dtype)
    for ii in xrange(numImages):
        frames[ii, :, :] = arrayList[ii][chopPx:imsize[1]-chopPx, chopPx:imsize[0]-chopPx]

    return frames



def makeDataset():
    random.seed(123)
    random.shuffle(dirNames)
    
    thisBatchNames = []
    thisBatchFrames = []
    counter = 0
    for dirName in dirNames:
        frames = loadImgs(os.path.join(dataDir, dirName))
        thisBatchNames.append(dirName)
        thisBatchFrames.append(frames)
        framesSoFar = sum([block.shape[0] for block in thisBatchFrames])
        if framesSoFar > minBatchSize:
            # dump to file
            with gzip.open(outputName % counter, 'wb') as ff:
                pickle.dump((thisBatchFrames, thisBatchNames), ff, pickle.HIGHEST_PROTOCOL)
            print 'Wrote:', outputName % counter
            thisBatchNames  = []
            thisBatchFrames = []
            counter += 1
    # Dump last batch
    if len(thisBatchFrames) > 0:
        with gzip.open(outputName % counter, 'wb') as ff:
            pickle.dump((thisBatchFrames, thisBatchNames), ff, pickle.HIGHEST_PROTOCOL)
        counter += 1
            
    print 'Wrote %d batches' % counter



def normalizeSaxeData(inputInt):
    ret = []
    for ii in range(len(inputInt)):
        normalized = array(inputInt[ii], copy = True, dtype = normalizationOutputDtype)
        normalized -= normalizationApproxMean
        normalized /= (max(normalizationApproxMean, 255-normalizationApproxMean) / normalizationMinMax)
        ret.append(normalized)
    return ret



def saveNormalizedDataset():
    ii = 0
    while True:
        try:
            with gzip.open(os.path.join(dataDir, 'batch_%02d.pkl.gz' % ii), 'rb') as ff:
                batchFramesInt, batchNames = pickle.load(ff)
        except IOError:
            if ii == 0:
                raise
            else:
                break
        print 'loaded batch %d: %s' % (ii, repr(batchNames))

        batchFrames = normalizeSaxeData(batchFramesInt)

        with gzip.open(os.path.join(dataDir, 'batch_%02d.normalized.pkl.gz' % ii), 'wb') as ff:
            pickle.dump((batchFrames, batchNames), ff, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dataDir, 'batch_%02d.normalized.pkl' % ii), 'wb') as ff:
            pickle.dump((batchFrames, batchNames), ff, pickle.HIGHEST_PROTOCOL)
        print '  saved normalized version'

        ii += 1



def main():
    #makeDataset()
    saveNormalizedDataset()



if __name__ == '__main__':
    main()
