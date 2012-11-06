#! /usr/bin/env python
#! /usr/local/bin/ipython --gui=wx

import os, pdb, gzip, sys
import subprocess
import re
import pp
import numpy
from util.dataLoaders import saveToFile
from numpy import *



SHAPE_VOXELS = 2000
THRESHOLD    = 0.1

def getTxtFiles(dataPath):
    ret = []
    pattern = re.compile('^[^_]+_[0-9]{5}_EXPORT_[0-9]+\.txt$')
    for root, dirs, files in os.walk(dataPath):
        for fil in files:
            if pattern.search(fil):
                filename = os.path.join(root, fil)
                #print filename,
                #print 'yes'
                ret.append(filename)
                if len(ret) % 1000 == 0:
                    print 'txtFiles so far:', len(ret)
            else:
                pass
                #print 'no'

    return ret



def txt2Mat(txtFile, shapeVoxels):
    with open(txtFile, 'r') as ff:
        lines = ff.readlines()
        assert lines[0].strip() == 'v1.0'        # Version
        assert lines[1].strip() == '0.1'         #  == THRESHOLD
        assert lines[2].strip() == '10 10 20'    #  == SHAPE_VOXELS
        mat = numpy.fromstring(lines[3], sep = ' ')
        assert mat.shape[0] == shapeVoxels
    return mat



def efSample(availableTxtFiles, fileFilter = None, seed = 0, Nsamples = 10):
    txtFiles = list(availableTxtFiles)
    nTxtFiles = len(txtFiles)
    random.seed(seed)

    #if fileFilter:
    #    lenFilt = len(fileFilter)
    #    filteredFiles = []
    #    for txtFile in txtFiles:
    #        path, filename = os.path.split(txtFile)
    #        if filename[:lenFilt] == fileFilter:
    #            filteredFiles.append(txtFile)
    #    txtFiles = filteredFiles
    #    nTxtFiles = len(txtFiles)
    #    print 'Filtered using', fileFilter, 'to', nTxtFiles, 'txt files'

    if Nsamples > nTxtFiles:
        raise Exception('Did not find enough txt files (%d < requested %d)' % (nTxtFiles, Nsamples))

    print 'Choosing', Nsamples, 'random files'

    random.shuffle(txtFiles)
    txtFiles = txtFiles[:Nsamples]
    # Convert 'aaaix0tl1w_00000_EXPORT_4.txt' -> ('aaaix0tl1w', 0, 4)
    labels = []
    for txtFile in txtFiles:
        path, filename = os.path.split(txtFile)
        runId, genSerial, junk, orgId = filename[:-4].split('_')
        labels.append((runId, int(genSerial), int(orgId)))

    job_server = pp.Server(ncpus=3)
    jobs = []
    
    USE_PP = False
    data = zeros((Nsamples, SHAPE_VOXELS))
    for ii, txtFile in enumerate(txtFiles):
        if USE_PP:
            jobs.append((ii, txtFile,
                         job_server.submit(txt2Mat,
                                           (txtFile, SHAPE_VOXELS),
                                           modules=('numpy',),
                                           ))
                        )
            #print 'started', ii
        else:
            data[ii,:] = txt2Mat(txtFile, SHAPE_VOXELS)
            #print 'done with', txtFile

    if USE_PP:
        for ii, txtFile, job in jobs:
            #print ii, txtFile,
            sys.stdout.flush()
            data[ii,:] = job()
            #print 'done'
            #print ii, txtFile, results, 'done'
            if ii % 100 == 0:
                print 'Finished %d/%d jobs' % (ii, len(jobs))
            if ii % 10000 == 0:
                job_server.print_stats()

        print

        job_server.print_stats()

    return labels, data



def main():
    if len(sys.argv) <= 1:
        print 'Usage:\n    # process txt files into matrix of values.\n    %s path_to_directory_of_shapes' % (sys.argv[0])
        sys.exit(1)

    dataPath = sys.argv[1]

    txtFiles = getTxtFiles(dataPath)
    nTxtFiles = len(txtFiles)
    print 'Found', nTxtFiles, 'txt files'

    txtFiles.sort()

    trainFiles = []
    testFiles  = []
    for txtFile in txtFiles:
        path, filename = os.path.split(txtFile)
        # class 0 (train): '0, 1..., 9, a, b... h'
        # class 1 (test):  'i, j, ... z'
        label = 0 if filename[2] <= 'h' else 1
        #print filename, label
        if label == 0:
            trainFiles.append(txtFile)
        else:
            testFiles.append(txtFile)

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 50)
    saveToFile('../data/endlessforms/train_real_50.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_50.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 50)
    saveToFile('../data/endlessforms/train_real_50.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_50.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 500)
    saveToFile('../data/endlessforms/train_real_500.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_500.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 500)
    saveToFile('../data/endlessforms/train_real_500.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_500.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 5000)
    saveToFile('../data/endlessforms/train_real_5000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_5000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 5000)
    saveToFile('../data/endlessforms/train_real_5000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_5000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 50000)
    saveToFile('../data/endlessforms/train_real_50000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_50000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 50000)
    saveToFile('../data/endlessforms/train_real_50000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_50000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 100000)
    saveToFile('../data/endlessforms/train_real_100000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_100000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 100000)
    saveToFile('../data/endlessforms/train_real_100000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_100000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 200000)
    saveToFile('../data/endlessforms/train_real_200000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_200000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 200000)
    saveToFile('../data/endlessforms/train_real_200000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_200000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 300000)
    saveToFile('../data/endlessforms/train_real_300000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_300000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 300000)
    saveToFile('../data/endlessforms/train_real_300000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_300000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 400000)
    saveToFile('../data/endlessforms/train_real_400000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_400000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 400000)
    saveToFile('../data/endlessforms/train_real_400000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_400000.pkl.gz', (labels, data > THRESHOLD))

    labels, data = efSample(trainFiles, seed = 0, Nsamples = 500000)
    saveToFile('../data/endlessforms/train_real_500000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_500000.pkl.gz', (labels, data > THRESHOLD))
    labels, data = efSample(testFiles, seed = 0, Nsamples = 500000)
    saveToFile('../data/endlessforms/train_real_500000.pkl.gz',   (labels, data))
    saveToFile('../data/endlessforms/train_thresh_500000.pkl.gz', (labels, data > THRESHOLD))



if __name__ == '__main__':
    main()
