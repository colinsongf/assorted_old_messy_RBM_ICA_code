#! /usr/bin/env python

import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle
import ipdb
from util.dataLoaders import saveToFile
from makeUpsonRovio1 import getFilesIn



def randomSampleMatrix(filterNames, seed, color, Nw = 10, Nsamples = 10):
    '''color = True or False'''

    # reset the random seed before each sample
    random.seed(seed)

    files = getFilesIn('../data/upson_rovio_2/imgfiles')

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
    # for efficient loading and unloading of images into memory. Re-randomize before returing
    randomSamples = randomSamples[argsort(randomSamples[:,0]), :]
    
    if color:
        imageMatrix = zeros((Nsamples, Nw * Nw * 3), dtype = float32)
    else:
        imageMatrix = zeros((Nsamples, Nw * Nw), dtype = float32)
    
    imIdx = None
    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if imIdx != idx:
            #print 'loading', idx
            if color:
                im = Image.open(filteredFiles[idx])
            else:
                im = Image.open(filteredFiles[idx]).convert('L')
            imIdx = idx
            if size != im.size:
                raise Exception('Expected everything to be the same size but %s != %s' % (repr(size), repr(im.size)))
            #im.show()

        cropped = im.crop((jj, ii, jj + Nw, ii + Nw))
        # For color images, flattens to [ii_r ii_g ii_b ii+1_r ii+1_g ii+1_b ...]
        imageMatrix[count,:] = array(cropped.getdata()).flatten()
        #cropped.show()

    imageMatrix /= 255   # normalize to 0-1 range
    random.shuffle(imageMatrix)
    return imageMatrix



def main():
    '''Data:

    u2_backward_0_person u2_backward_1 u2_backward_2 u2_backward_3
    u2_forward_0_person u2_forward_1 u2_stationary_0_person
    u2_stationary_1 u2_strafe_r_0 u2_strafe_r_1 u2_strafe_r_2
    u2_strafe_r_3 u2_turn_r_0 u2_turn_r_1'''
    
    trainFilter = ['u2_backward_0_person',
                   'u2_backward_2',
                   'u2_forward_1',
                   'u2_stationary_0_person',
                   'u2_strafe_r_0',
                   'u2_strafe_r_2',
                   'u2_turn_r_0']

    testFilter  = ['u2_backward_1',
                   'u2_backward_3',
                   'u2_forward_0_person',
                   'u2_stationary_1',
                   'u2_strafe_r_1',
                   'u2_strafe_r_3',
                   'u2_turn_r_1']

    sav = saveToFile
    rsm = randomSampleMatrix
    flnm = '../data/upson_rovio_2/%s.pkl.gz'

    # Random Seeds: use the window width, so 50000 data set
    # contains the 50 data set, and so bw and color sample the same
    # patches. Train vs. test are different though, due to the
    # negative.

    big = 123456
    
    # grayscale
    for Nw in [2, 4, 8, 10, 15, 20, 28, 30, 40]:
        sav(flnm % ('train_%02d_50_1c' % Nw),    rsm(trainFilter,    Nw, False, Nw = Nw, Nsamples = 50))
        sav(flnm % ('test_%02d_50_1c' % Nw),     rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50))
        sav(flnm % ('train_%02d_50000_1c' % Nw), rsm(trainFilter,    Nw, False, Nw = Nw, Nsamples = 50000))
        sav(flnm % ('test_%02d_50000_1c' % Nw),  rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50000))

    # color (same seeds as grayscale so same patches are selected)
    #    Runs out of memory on largest: 40_50000_3c (test or train)
    for Nw in [2, 4, 8, 10, 15, 20, 28, 30, 40]:
        sav(flnm % ('train_%02d_50_3c' % Nw),    rsm(trainFilter,    Nw, True, Nw = Nw, Nsamples = 50))
        sav(flnm % ('test_%02d_50_3c' % Nw),     rsm(testFilter, big-Nw, True, Nw = Nw, Nsamples = 50))
        sav(flnm % ('train_%02d_50000_3c' % Nw), rsm(trainFilter,    Nw, True, Nw = Nw, Nsamples = 50000))
        sav(flnm % ('test_%02d_50000_3c' % Nw),  rsm(testFilter, big-Nw, True, Nw = Nw, Nsamples = 50000))



if __name__ == '__main__':
    main()
