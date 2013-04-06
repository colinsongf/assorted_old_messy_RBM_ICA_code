#! /usr/bin/env python

import os, pdb, gzip
from PIL import Image
from numpy import *
import cPickle as pickle
import ipdb as pdb
from util.dataLoaders import saveToFile
from makeUpsonRovio1 import getFilesIn


trainFilter = ['u3_backward_0_person',
               'u3_backward_2',
               'u3_forward_1',
               'u3_stationary_0_person',
               'u3_strafe_r_0',
               'u3_strafe_r_2',
               'u3_turn_r_0',
               'u3_green_circle_close_1',
               'u3_green_circle_far_1',
               'u3_green_star_close_1',
               'u3_green_star_far_1',
               'u3_red_circle_close_1',
               'u3_red_circle_far_1',
               'u3_red_star_close_1',
               'u3_red_star_far_1',
               ]

testFilter  = ['u3_backward_1',
               'u3_backward_3',
               'u3_forward_0_person',
               'u3_stationary_1',
               'u3_strafe_r_1',
               'u3_strafe_r_3',
               'u3_turn_r_1',
               'u3_green_circle_close_2',
               'u3_green_circle_far_2',
               'u3_green_circle_far_high',
               'u3_green_star_close_2',
               'u3_green_star_far_2',
               'u3_green_star_far_high',
               'u3_red_circle_close_2',
               'u3_red_circle_far_2',
               'u3_red_circle_far_high',
               'u3_red_star_close_2',
               'u3_red_star_far_2',
               'u3_red_star_far_high'
               ]

exploreFilter = ['u3_all_shapes_tour',
                 'u3_jason_laptop',
                 ]

labelStrings = ['circle',
                'square',
                'red',
                'green',
                'close',
                'far',
                'high',
                'backward',
                'foreward ',
                'stationary',
                'strafe',
                'turn',
                'person']




def randomSampleMatrixWithLabels(filterNames, seed, color, Nw = 10, Nsamples = 10,
                                 imgDirectory = '../data/upson_rovio_3/imgfiles'):
    '''color = True or False'''

    # reset the random seed before each sample
    random.seed(seed)

    files = getFilesIn(imgDirectory)

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
    labelMatrix = zeros((Nsamples, len(labelStrings)))
    
    imIdx = None
    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if imIdx != idx:
            #print 'loading', idx
            if color:
                im = Image.open(filteredFiles[idx])
            else:
                im = Image.open(filteredFiles[idx]).convert('L')
            filename = os.path.basename(filteredFiles[idx])
            labels = array([st in filename for st in labelStrings], dtype=int)
            imIdx = idx
            if size != im.size:
                raise Exception('Expected everything to be the same size but %s != %s' % (repr(size), repr(im.size)))
            #im.show()

        cropped = im.crop((jj, ii, jj + Nw, ii + Nw))
        # For color images, flattens to [ii_r ii_g ii_b ii+1_r ii+1_g ii+1_b ...]
        imageMatrix[count,:] = array(cropped.getdata()).flatten()
        labelMatrix[count,:] = labels[:]
        #cropped.show()

    imageMatrix /= 255   # normalize to 0-1 range

    # shuffle both matrices together
    shufIdx = random.permutation(range(Nsamples))
    imageMatrix = imageMatrix[shufIdx,:]
    labelMatrix = labelMatrix[shufIdx,:]

    return imageMatrix, labelMatrix, labelStrings



def main():
    sav = saveToFile
    rsm = randomSampleMatrixWithLabels
    flnm = '../data/upson_rovio_3/%s.pkl.gz'

    # Random Seeds: use the window width, so 50000 data set
    # contains the 50 data set, and so bw and color sample the same
    # patches. Train vs. test are different though, due to the
    # negative.

    big = 123456
    
    # grayscale
    #for Nw in [2, 4, 8, 10, 15, 20, 28, 30, 40]:
    for Nw in [2, 3, 4, 6, 8, 10, 15, 20, 25, 28, 30, 40]:
        sav(flnm % ('train_%02d_50_1c' % Nw),    rsm(trainFilter,    Nw, False, Nw = Nw, Nsamples = 50))
        sav(flnm % ('test_%02d_50_1c' % Nw),     rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50))
        sav(flnm % ('explore_%02d_50_1c' % Nw),     rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50))
        sav(flnm % ('train_%02d_50000_1c' % Nw), rsm(trainFilter,    Nw, False, Nw = Nw, Nsamples = 50000))
        sav(flnm % ('test_%02d_50000_1c' % Nw),  rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50000))
        sav(flnm % ('explore_%02d_50000_1c' % Nw),  rsm(testFilter, big-Nw, False, Nw = Nw, Nsamples = 50000))

    # color (same seeds as grayscale so same patches are selected)
    #    Runs out of memory on largest: 40_50000_3c (test or train)
    #for Nw in [2, 4, 8, 10, 15, 20, 28, 30, 40]:
    for Nw in [2, 3, 4, 6, 8, 10, 15, 20, 25, 28, 30, 40]:
        sav(flnm % ('train_%02d_50_3c' % Nw),    rsm(trainFilter,    Nw, True, Nw = Nw, Nsamples = 50))
        sav(flnm % ('test_%02d_50_3c' % Nw),     rsm(testFilter, big-Nw, True, Nw = Nw, Nsamples = 50))
        sav(flnm % ('train_%02d_50000_3c' % Nw), rsm(trainFilter,    Nw, True, Nw = Nw, Nsamples = 50000))
        sav(flnm % ('test_%02d_50000_3c' % Nw),  rsm(testFilter, big-Nw, True, Nw = Nw, Nsamples = 50000))



if __name__ == '__main__':
    main()
