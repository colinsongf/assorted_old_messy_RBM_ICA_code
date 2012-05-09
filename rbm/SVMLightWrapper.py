#! /usr/bin/env python

from numpy import array, random, sum, zeros, mean
import subprocess as sp
import os
import pdb


def writeToFile(filename, X, y = None):
    '''Writes X and y (optiona) to file in the format SVM Light
    expects.'''

    if y is None:
        y = zeros(X.shape[0])
        
    assert(X.shape[0] == len(y))

    print 'Got %d data points of dim %d' % (len(y), X.shape[1])

    # HACKED!
    ff = open(filename, 'w')
    for ii in range(X.shape[0]):
        ff.write('%d ' % y[ii])
        st = ' '.join(['%d:%d' % (jj+1,X[ii,jj]) for jj in range(X.shape[1]) if X[ii,jj] != 0])
        ff.write(st)
        ff.write('\n')
    ff.close()



def getTmpFile():
    return '/tmp/svmwrap_%s' % ''.join(['%d' % x for x in 10*random.rand(10)])



class SVMLightWrapper(object):
    '''Wrapper for SVM Light binary'''

    def __init__(self, kernelType = 0, C = 1, z = 'c'):
        self.kernelType   = kernelType
        self.C            = C
        self.z            = z
        self.trainExec    = './svm_learn'
        self.classifyExec = './svm_classify'

        self.modelFile    = None

    def train(self, X, y):
        '''./svm_learn -t 0 -z r -c .1 simple_train model_simple'''

        if self.isTrained():
            print 'Warning: overwriting previous training.'

        if self.z == 'c' and y.dtype == bool:
            y = y * 2 - 1
        
        trainDataFile = getTmpFile()
        writeToFile(trainDataFile, X, y)

        self.modelFile = getTmpFile()
        
        proc = sp.Popen((self.trainExec,
                         '-t', '%d' % self.kernelType,
                         '-z', self.z,
                         '-c', '%f' % self.C,
                         trainDataFile, self.modelFile),
                        stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)
        out,err = proc.communicate()
        code = proc.wait()

        os.unlink(trainDataFile)
        #print trainDataFile
        #pdb.set_trace()


    def predict(self, X):
        '''./svm_classify simple_test model_simple predictions'''

        if not self.isTrained():
            raise Exception('Must train first')
        
        numExamples = X.shape[0]

        testDataFile = getTmpFile()
        writeToFile(testDataFile, X)

        predictionFile = getTmpFile()
        
        proc = sp.Popen((self.classifyExec,
                         testDataFile, self.modelFile,
                         predictionFile),
                        stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)
        out,err = proc.communicate()
        code = proc.wait()

        ret = zeros(numExamples)
        for ii,line in enumerate(open(predictionFile, 'r')):
            ret[ii] = float(line.strip())
        if ii + 1 != numExamples:
            raise Exception('Expected %d lines but got %d' % (numExamples, ii+1))

        os.unlink(testDataFile)
        os.unlink(predictionFile)

        return ret

    def isTrained(self):
        return self.modelFile is not None



def main():
    # random training data
    randData = random.rand(100,11)
    trainX = randData[:,0:10]
    mn = mean(trainX) * 10
    trainy = sum(randData, 1) > mn

    # random test data
    randData = random.rand(100,11)
    testX = randData[:,0:10]
    testy = sum(randData, 1) > mn

    svr = SVMLightWrapper()
    svr.train(trainX, trainy)
    predy = svr.predict(testX)

    print testy
    print predy
    print predy > 0



if __name__ == '__main__':
    main()
