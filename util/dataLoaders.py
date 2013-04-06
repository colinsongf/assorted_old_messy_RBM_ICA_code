#! /usr/bin/env python

import cPickle as pickle
import numpy
from numpy import array, zeros, reshape
import gzip
import cPickle as pickle
import ipdb



def loadCifarData(cifarDirectory):
    ''' Loads the CIFAR-10 dataset.'''

    if cifarDirectory[-1] != '/':
        cifarDirectory += '/'

    fileNames = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4','data_batch_5', 'test_batch']

    dicts = []
    for fileName in fileNames:
        with open(cifarDirectory + fileName, 'rb') as ff:
            dicts.append(pickle.load(ff))

    batchMeta    = dicts[0]
    trainBatches = dicts[1:6]
    testBatch    = dicts[6]

    train_set_x = zeros((50000, 1024*3), dtype=numpy.float32)
    train_set_y = zeros(50000, dtype=numpy.int8)
    for ii, trainBatch in enumerate(trainBatches):
        train_set_x[ii*10000:(ii+1)*10000, :] = trainBatch['data']
        train_set_y[ii*10000:(ii+1)*10000]    = trainBatch['labels']
    test_set_x = array(testBatch['data'], dtype=numpy.float32)
    test_set_y = array(testBatch['labels'], dtype=numpy.int8)

    classNames = batchMeta['label_names']

    # no validation set,
    return [[train_set_x, train_set_y], [array([]), None], [test_set_x, test_set_y]], classNames



def loadCifarDataMonochrome(cifarDirectory):
    ''' Loads the CIFAR-10 dataset but makes it monochrome'''

    datasets, classNames = loadCifarData(cifarDirectory)

    train_set_x_mono = zeros((50000, 1024), dtype=numpy.float32)
    test_set_x_mono = zeros((10000, 1024), dtype=numpy.float32)
    for ii in range(3):
        train_set_x_mono += datasets[0][0][:, ii*1024:(ii+1)*1024]
        test_set_x_mono  += datasets[2][0][:, ii*1024:(ii+1)*1024]
    train_set_x_mono /= 3
    test_set_x_mono /= 3

    # no validation set,
    return [[train_set_x_mono, datasets[0][1]], datasets[1], [test_set_x_mono, datasets[2][1]]], classNames



def loadCifarDataSubsets(cifarDirectory, size, topLeftCoords):
    ''' Loads the CIFAR-10 dataset but chops it up into
    len(topLeftCoords) windows of size SIZE, where this first window
    is anchored at (ii,jj) = topLeftCoords[0], etc...'''

    datasets, classNames = loadCifarData(cifarDirectory)

    print 'Abandoned for now... pick up later, maybe'
    ipdb.set_trace()



def saveToFile(filename, obj, quiet = False):
    ff = gzip.open(filename, 'wb')
    pickle.dump(obj, ff, protocol = -1)
    if not quiet:
        print 'saved to', filename
    ff.close()



def loadFromPklGz(filename):
    with gzip.open(filename, 'rb') as ff:
        ret = pickle.load(ff)
    return ret



def loadAtariData(filename):
    '''Loads Atari Data'''

    data = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    return data



def loadUpsonData(filename):
    '''Loads Upson Data'''

    data = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    return data



def loadUpsonData3(filename):
    '''Loads Upson Data from the upson_rovio_3 dataset (with labels)'''

    data,labels,labelStrings = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    labels = labels.T
    return data, labels, labelStrings



def loadRandomData(filename):
    '''Loads Random Data'''

    data = loadFromPklGz(filename)
    return data

