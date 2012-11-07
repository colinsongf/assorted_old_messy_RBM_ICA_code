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
                if len(ret) % 10000 == 0:
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



def efSample(availableTxtFiles, saveLocation, fileFilter = None, seed = 0, Nsamples = 10, Nsplits = 1):
    '''saveLocation like "../data/endlessforms/train_%s_%d_%d.pkl.gz" (real, size, serial)'''

    allTxtFiles = list(availableTxtFiles)
    nTxtFiles = len(allTxtFiles)
    random.seed(seed)
    random.shuffle(allTxtFiles)

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

    Ntotal = Nsamples * Nsplits
    if Ntotal > nTxtFiles:
        raise Exception('Did not find enough txt files (%d < requested %d)' % (nTxtFiles, Ntotal))

    print 'Choosing', Ntotal, 'random files total (%d per file x %d files)' % (Nsamples, Nsplits)

    for splitIdx in range(Nsplits):
        #print 'Split', splitIdx

        txtFiles = allTxtFiles[(splitIdx*Nsamples):((splitIdx+1)*Nsamples)]
        labels = []
        for txtFile in txtFiles:
            # Convert 'aaaix0tl1w_00000_EXPORT_4.txt' -> ('aaaix0tl1w', 0, 4)
            path, filename = os.path.split(txtFile)
            runId, genSerial, junk, orgId = filename[:-4].split('_')
            labels.append((runId, int(genSerial), int(orgId)))

        USE_PP = False

        if USE_PP:
            job_server = pp.Server(ncpus = 20)
            jobs = []

        data = zeros((Nsamples, SHAPE_VOXELS), dtype = 'float32')

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
                #sys.stdout.flush()
                data[ii,:] = job()
                #print 'done'
                #print ii, txtFile, results, 'done'
                if ii % 100 == 0:
                    print 'Finished %d/%d jobs' % (ii, len(jobs))
                if ii % 10000 == 0:
                    job_server.print_stats()

            print

            job_server.print_stats()

        saveToFile(saveLocation % ('real', Nsamples, splitIdx), (labels, data))
        saveToFile(saveLocation % ('bool', Nsamples, splitIdx), (labels, data > THRESHOLD))



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

    efSample(trainFiles, '../data/endlessforms/train_%s_%d_%d.pkl.gz', seed = 0, Nsamples = 50,    Nsplits = 10)
    efSample(testFiles,  '../data/endlessforms/test_%s_%d_%d.pkl.gz',  seed = 0, Nsamples = 50,    Nsplits = 10)

    efSample(trainFiles, '../data/endlessforms/train_%s_%d_%d.pkl.gz', seed = 0, Nsamples = 500,   Nsplits = 10)
    efSample(testFiles,  '../data/endlessforms/test_%s_%d_%d.pkl.gz',  seed = 0, Nsamples = 500,   Nsplits = 10)

    efSample(trainFiles, '../data/endlessforms/train_%s_%d_%d.pkl.gz', seed = 0, Nsamples = 5000,  Nsplits = 10)
    efSample(testFiles,  '../data/endlessforms/test_%s_%d_%d.pkl.gz',  seed = 0, Nsamples = 5000,  Nsplits = 10)

    efSample(trainFiles, '../data/endlessforms/train_%s_%d_%d.pkl.gz', seed = 0, Nsamples = 50000, Nsplits = 10)
    efSample(testFiles,  '../data/endlessforms/test_%s_%d_%d.pkl.gz',  seed = 0, Nsamples = 50000, Nsplits = 10)



if __name__ == '__main__':
    main()
