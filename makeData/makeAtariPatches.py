#! /usr/bin/env python


import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle

from rbm.utils import DuckStruct
from util.dataLoaders import saveToFile



def getFilesIn(dir):
    fileList = []
    for dd,junk,files in os.walk(dir, followlinks=True):
        for file in files:
            fileList.append(os.path.join(dd, file))
    return fileList



def randomSampleMatrix(path, filterNames, Nw = 10, Nsamples = 10, color = False):
    files = getFilesIn(path)

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
    randomSamples = vstack((random.randint(0, Nimages, Nsamples),
                            random.randint(0, maxI+1, Nsamples),
                            random.randint(0, maxJ+1, Nsamples))).T
    sortIdx = argsort(randomSamples[:,0])
    randomSamples = randomSamples[sortIdx,:]   # for efficient loading and unloading of images into memory. Re-randomize before returing

    nChannels = 3 if color else 1
    imageMatrix = zeros((Nsamples, Nw * Nw * nChannels), dtype = float32)
    
    imIdx = None
    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if imIdx != idx:
            #print 'loading', idx
            im = Image.open(filteredFiles[idx])
            if not color:
                im = im.convert('L')
            imIdx = idx
            if size != im.size:
                raise Exception('Expected everything to be the same size but %s != %s' % (repr(size), repr(im.size)))
            #im.show()
        cropped = im.crop((jj, ii, jj + Nw, ii + Nw))
        imageMatrix[count,:] = asarray(cropped).flatten()
        #cropped.show()

    imageMatrix /= 255   # normalize to 0-1 range
    random.shuffle(imageMatrix)
    return imageMatrix



def main():
    datasets = []

    datasets.append(DuckStruct(name = '../data/atari/mspacman_',
                               path = '../data/atari/mspacman',
                               trainFilter = ['frame_000001', 'frame_000002', 'frame_000003'],
                               testFilter  = ['frame_000004', 'frame_000005', 'frame_000006']))

    datasets.append(DuckStruct(name = '../data/atari/space_invaders_',
                               path = '../data/atari/space_invaders',
                               trainFilter = ['frame_0000%02d' % x for x in range(0,6)],
                               testFilter  = ['frame_0000%02d' % x for x in range(6,12)]))

    for dataset in datasets:
        random.seed(0)

        name, path, trainFilter, testFilter = dataset.name, dataset.path, dataset.trainFilter, dataset.testFilter

        print 'Looking for data in: ', path

        #test
        #saveToFile(pathToData + 'junk.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 15, Nsamples = 50000, color = True))
        #sys.exit(1)

        # monochrome
        saveToFile(name + 'train_02_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 02, Nsamples = 50, color = False))
        saveToFile(name + 'test_02_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 02, Nsamples = 50, color = False))
        saveToFile(name + 'train_02_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 02, Nsamples = 50000, color = False))
        saveToFile(name + 'test_02_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 02, Nsamples = 50000, color = False))

        saveToFile(name + 'train_03_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 03, Nsamples = 50, color = False))
        saveToFile(name + 'test_03_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 03, Nsamples = 50, color = False))
        saveToFile(name + 'train_03_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 03, Nsamples = 50000, color = False))
        saveToFile(name + 'test_03_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 03, Nsamples = 50000, color = False))

        saveToFile(name + 'train_04_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 04, Nsamples = 50, color = False))
        saveToFile(name + 'test_04_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 04, Nsamples = 50, color = False))
        saveToFile(name + 'train_04_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 04, Nsamples = 50000, color = False))
        saveToFile(name + 'test_04_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 04, Nsamples = 50000, color = False))

        saveToFile(name + 'train_06_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 06, Nsamples = 50, color = False))
        saveToFile(name + 'test_06_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 06, Nsamples = 50, color = False))
        saveToFile(name + 'train_06_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 06, Nsamples = 50000, color = False))
        saveToFile(name + 'test_06_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 06, Nsamples = 50000, color = False))

        saveToFile(name + 'train_10_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 10, Nsamples = 50, color = False))
        saveToFile(name + 'test_10_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 10, Nsamples = 50, color = False))
        saveToFile(name + 'train_10_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 10, Nsamples = 50000, color = False))
        saveToFile(name + 'test_10_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 10, Nsamples = 50000, color = False))

        saveToFile(name + 'train_15_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 15, Nsamples = 50, color = False))
        saveToFile(name + 'test_15_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 15, Nsamples = 50, color = False))
        saveToFile(name + 'train_15_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 15, Nsamples = 50000, color = False))
        saveToFile(name + 'test_15_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 15, Nsamples = 50000, color = False))

        saveToFile(name + 'train_20_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 20, Nsamples = 50, color = False))
        saveToFile(name + 'test_20_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 20, Nsamples = 50, color = False))
        saveToFile(name + 'train_20_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 20, Nsamples = 50000, color = False))
        saveToFile(name + 'test_20_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 20, Nsamples = 50000, color = False))

        saveToFile(name + 'train_25_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 25, Nsamples = 50, color = False))
        saveToFile(name + 'test_25_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 25, Nsamples = 50, color = False))
        saveToFile(name + 'train_25_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 25, Nsamples = 50000, color = False))
        saveToFile(name + 'test_25_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 25, Nsamples = 50000, color = False))

        saveToFile(name + 'train_28_50_1c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 28, Nsamples = 50, color = False))
        saveToFile(name + 'test_28_50_1c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 28, Nsamples = 50, color = False))
        saveToFile(name + 'train_28_50000_1c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 28, Nsamples = 50000, color = False))
        saveToFile(name + 'test_28_50000_1c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 28, Nsamples = 50000, color = False))

        # color
        saveToFile(name + 'train_02_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 02, Nsamples = 50, color = True))
        saveToFile(name + 'test_02_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 02, Nsamples = 50, color = True))
        saveToFile(name + 'train_02_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 02, Nsamples = 50000, color = True))
        saveToFile(name + 'test_02_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 02, Nsamples = 50000, color = True))

        saveToFile(name + 'train_03_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 03, Nsamples = 50, color = True))
        saveToFile(name + 'test_03_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 03, Nsamples = 50, color = True))
        saveToFile(name + 'train_03_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 03, Nsamples = 50000, color = True))
        saveToFile(name + 'test_03_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 03, Nsamples = 50000, color = True))

        saveToFile(name + 'train_04_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 04, Nsamples = 50, color = True))
        saveToFile(name + 'test_04_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 04, Nsamples = 50, color = True))
        saveToFile(name + 'train_04_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 04, Nsamples = 50000, color = True))
        saveToFile(name + 'test_04_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 04, Nsamples = 50000, color = True))

        saveToFile(name + 'train_06_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 06, Nsamples = 50, color = True))
        saveToFile(name + 'test_06_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 06, Nsamples = 50, color = True))
        saveToFile(name + 'train_06_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 06, Nsamples = 50000, color = True))
        saveToFile(name + 'test_06_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 06, Nsamples = 50000, color = True))

        saveToFile(name + 'train_10_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 10, Nsamples = 50, color = True))
        saveToFile(name + 'test_10_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 10, Nsamples = 50, color = True))
        saveToFile(name + 'train_10_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 10, Nsamples = 50000, color = True))
        saveToFile(name + 'test_10_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 10, Nsamples = 50000, color = True))

        saveToFile(name + 'train_15_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 15, Nsamples = 50, color = True))
        saveToFile(name + 'test_15_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 15, Nsamples = 50, color = True))
        saveToFile(name + 'train_15_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 15, Nsamples = 50000, color = True))
        saveToFile(name + 'test_15_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 15, Nsamples = 50000, color = True))

        saveToFile(name + 'train_20_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 20, Nsamples = 50, color = True))
        saveToFile(name + 'test_20_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 20, Nsamples = 50, color = True))
        saveToFile(name + 'train_20_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 20, Nsamples = 50000, color = True))
        saveToFile(name + 'test_20_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 20, Nsamples = 50000, color = True))

        saveToFile(name + 'train_25_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 25, Nsamples = 50, color = True))
        saveToFile(name + 'test_25_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 25, Nsamples = 50, color = True))
        saveToFile(name + 'train_25_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 25, Nsamples = 50000, color = True))
        saveToFile(name + 'test_25_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 25, Nsamples = 50000, color = True))

        saveToFile(name + 'train_28_50_3c.pkl.gz',    randomSampleMatrix(path, trainFilter, Nw = 28, Nsamples = 50, color = True))
        saveToFile(name + 'test_28_50_3c.pkl.gz',     randomSampleMatrix(path, testFilter,  Nw = 28, Nsamples = 50, color = True))
        saveToFile(name + 'train_28_50000_3c.pkl.gz', randomSampleMatrix(path, trainFilter, Nw = 28, Nsamples = 50000, color = True))
        saveToFile(name + 'test_28_50000_3c.pkl.gz',  randomSampleMatrix(path, testFilter,  Nw = 28, Nsamples = 50000, color = True))



if __name__ == '__main__':
    main()
