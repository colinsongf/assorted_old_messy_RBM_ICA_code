#! /usr/bin/env python

import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle
from util.dataLoaders import saveToFile



def getFilesIn(dir):
    fileList = []
    for dd,junk,files in os.walk(dir, followlinks=True):
        for file in files:
            fileList.append(os.path.join(dd, file))
    return fileList



def randomSampleMatrix(filterNames, Nw = 10, Nsamples = 10):
    files = getFilesIn('../data/upson_rovio_1')

    filteredFiles = []
    for filterName in filterNames:
        filteredFiles += filter(lambda x : filterName in x, files)

    filteredFiles = list(set(filteredFiles))
    filteredFiles.sort()

    Nimages = len(filteredFiles)
    if Nimages == 0:
        raise Exception('Nimages == 0, maybe try running from a different directory?')
    
    im = Image.open(filteredFiles[0])
    size = im.size

    # select random windows
    maxJ = size[0] - Nw
    maxI = size[1] - Nw
    raise Exception('fix this next line!')
    randomSamples = vstack((random.randint(0, Nw, Nsamples),      # <-- should be Nimages
                            random.randint(0, maxI+1, Nsamples),
                            random.randint(0, maxJ+1, Nsamples))).T
    raise Exception('fix this next line (NOT row sort)!')
    randomSamples.sort(0)   # for efficient loading and unloading of images into memory. Re-randomize before returing
    
    imageMatrix = zeros((Nsamples, Nw * Nw), dtype = float32)
    
    imIdx = None
    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if imIdx != idx:
            #print 'loading', idx
            im = Image.open(filteredFiles[idx]).convert('L')
            imIdx = idx
            if size != im.size:
                raise Exception('Expected everything to be the same size but %s != %s' % (repr(size), repr(im.size)))
            #im.show()
        cropped = im.crop((jj, ii, jj + Nw, ii + Nw))
        imageMatrix[count,:] = cropped.getdata()
        #cropped.show()

    imageMatrix /= 255   # normalize to 0-1 range
    random.shuffle(imageMatrix)
    return imageMatrix



def main():
    random.seed(0)

    trainFilter = ['image-2534', 'image-2535']
    testFilter  = ['image-2545']

    saveToFile('../data/upson_rovio_1/train_02_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 02, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/test_02_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 02, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/train_02_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 02, Nsamples = 50000))
    saveToFile('../data/upson_rovio_1/test_02_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 02, Nsamples = 50000))

    saveToFile('../data/upson_rovio_1/train_04_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 04, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/test_04_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 04, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/train_04_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 04, Nsamples = 50000))
    saveToFile('../data/upson_rovio_1/test_04_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 04, Nsamples = 50000))

    #saveToFile('../data/upson_rovio_1/train_10_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 10, Nsamples = 50))
    #saveToFile('../data/upson_rovio_1/test_10_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 10, Nsamples = 50))
    #saveToFile('../data/upson_rovio_1/train_10_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 10, Nsamples = 50000))
    #saveToFile('../data/upson_rovio_1/test_10_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 10, Nsamples = 50000))

    saveToFile('../data/upson_rovio_1/train_15_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 15, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/test_15_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 15, Nsamples = 50))
    saveToFile('../data/upson_rovio_1/train_15_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 15, Nsamples = 50000))
    saveToFile('../data/upson_rovio_1/test_15_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 15, Nsamples = 50000))
    
    #saveToFile('../data/upson_rovio_1/train_28_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 28, Nsamples = 50))
    #saveToFile('../data/upson_rovio_1/test_28_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 28, Nsamples = 50))
    #saveToFile('../data/upson_rovio_1/train_28_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 28, Nsamples = 50000))
    #saveToFile('../data/upson_rovio_1/test_28_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    main()
