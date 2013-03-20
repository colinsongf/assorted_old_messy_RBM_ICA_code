#! /usr/bin/env ipythonpl

import pdb
from numpy import array, dot, random, linalg, sqrt, asarray, cov, eye, sum, hstack
from numpy.linalg import norm

from pca import testPca



class PCA_SVD(object):
    def __init__(self, xx):
        '''
        PROBABLY DO NOT USE THIS VERSION
        
        Inspired by PCA in matplotlib.mlab

        Compute the SVD of a dataset and stores the mean, dimsigma,
        and SVD for the data.  Use toPC and fromPC to project the data
        onto a reduced set of dimensions and back. This version does
        not use the covariance matrix, but takes the SVD of the raw /
        raw centered data.

        Inputs:

          *xx*: a numobservations x numdims array

        Attrs:

          *nn*, *mm*: the dimensions of xx

          *mu* : a numdims array of means of xx

          *sigma* : a numdims array of atandard deviation of xx

          *var* : the average amount of variance of each of the principal components

          *std* : sqrt of var

          *fracVar* : the fractional amount of variance from each principal component

          *fracStd* : sqrt of fracVar
        '''

        self.nn, self.mm = xx.shape
        if self.nn < self.mm:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.mu    = xx.mean(axis=0)
        self.sigma = xx.std(axis=0)

        self.centeredXX  = self.center(xx, normalized = True)

        UU, ss, self.Vh = linalg.svd(self.centeredXX, full_matrices=False)

        self.var = ss**2 / float(self.nn)
        self.std = sqrt(self.var)
        self.fracVar = self.var / self.var.sum()
        self.fracStd = self.std / self.std.sum()


    def pc(self):
        return self.Vh.T


    def toPC(self, xx, numDims = None):
        '''Center the xx and project it onto the principle components'''

        xx = asarray(xx)

        if xx.shape[-1] != self.mm:
            raise ValueError('Expected an array with dims[-1] == %d' % self.mm)

        if numDims is None:
            numDims = xx.shape[-1]
        
        centered = self.center(xx, normalized = True)
        pc = dot(centered, self.Vh[0:numDims,:].T) / self.std[0:numDims]   # more efficient
        
        return pc


    def fromPC(self, pc):
        '''Project the given principle components back to the original space and uncenter them.'''

        pc = asarray(pc)

        numDims = pc.shape[1]
        
        centered = dot(pc * self.std[0:numDims], self.Vh[0:numDims,:])
        xx = self.uncenter(centered, normalized = True)

        return xx


    def center(self, xx, normalized = False):
        '''Center the data using the mean and sigma from the training set'''

        if normalized:
            return (xx - self.mu) / self.sigma
        else:
            return xx - self.mu


    def uncenter(self, cc, normalized = False):
        '''Undo the operation of center'''

        if normalized:
            return (cc * self.sigma) + self.mu
        else:
            return cc + self.mu



if __name__ == '__main__':
    print 'New tests + old version of PCA_SVD probably do not play well together'
    testPca(PcaClass = PCA_SVD)
    raw_input('push enter to exit')
