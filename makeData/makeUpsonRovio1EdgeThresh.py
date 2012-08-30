#! /usr/bin/env python


import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle



def getFilesIn(dir):
    fileList = []
    for dd,junk,files in os.walk(dir, followlinks=True):
        for file in files:
            fileList.append(os.path.join(dd, file))
    return fileList



def randomSampleMatrix(filterNames, Nw = 10, Nsamples = 10):
    files = getFilesIn('../data/upson_rovio_1_edge_thresh')

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
    randomSamples = vstack((random.randint(0, Nw, Nsamples),
                            random.randint(0, maxI+1, Nsamples),
                            random.randint(0, maxJ+1, Nsamples))).T
    raise Exception('fix this next line!')
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



def saveToFile(filename, obj):
    ff = gzip.open(filename, 'wb')
    pickle.dump(obj, ff)
    print 'saved to', filename
    ff.close()



def main():
    random.seed(0)

    trainFilter = ['image-2534', 'image-2535']
    testFilter  = ['image-2545']

    saveToFile('../data/upson_rovio_1_edge_thresh/train_10_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 10, Nsamples = 50))
    saveToFile('../data/upson_rovio_1_edge_thresh/test_10_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 10, Nsamples = 50))
    saveToFile('../data/upson_rovio_1_edge_thresh/train_10_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 10, Nsamples = 50000))
    saveToFile('../data/upson_rovio_1_edge_thresh/test_10_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 10, Nsamples = 50000))

    saveToFile('../data/upson_rovio_1_edge_thresh/train_28_50.pkl.gz',    randomSampleMatrix(trainFilter, Nw = 28, Nsamples = 50))
    saveToFile('../data/upson_rovio_1_edge_thresh/test_28_50.pkl.gz',     randomSampleMatrix(testFilter,  Nw = 28, Nsamples = 50))
    saveToFile('../data/upson_rovio_1_edge_thresh/train_28_50000.pkl.gz', randomSampleMatrix(trainFilter, Nw = 28, Nsamples = 50000))
    saveToFile('../data/upson_rovio_1_edge_thresh/test_28_50000.pkl.gz',  randomSampleMatrix(testFilter,  Nw = 28, Nsamples = 50000))



if __name__ == '__main__':
    main()
