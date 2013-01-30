#! /usr/bin/env ipythonpl

import pdb
import os
from scipy import stats
from matplotlib import pyplot
from numpy import *

from GitResultsManager import resman



class IndependentGKEDensity(object):
    '''Independent Gaussian Kernel Estimated Density.

    Models data (dimension x num samples) using independent GKEs per dimension.'''


    def __init__(self):
        self.ndim = -1
        self.kde = []


    def fit(self, data):
        '''Fit (or train) the model given the data.'''
        if self.isFit():
            raise Exception('Density is already fit')
        self.ndim = data.shape[0]
        self.kde = []
        for dim in xrange(self.ndim):
            self.kde.append(stats.gaussian_kde(data[dim,:]))


    def isFit(self):
        return self.ndim != -1


    def sample(self, number):
        '''Generate random samples from the model'''

        if not self.isFit():
            return Exception('Not yet fit')
        
        ret = zeros((self.ndim, number))
        for dim in xrange(self.ndim):
            ret[dim,:] = self.kde[dim].resample(number)
        return ret


    def pdf(self, data, perDim = False):
        '''Returns the probability of the given samples.

        When not using perDim = True, for dimension larger than 10 or
        so, logpdf is recommended for numerical stability.'''

        if not self.isFit():
            return Exception('Not yet fit')
        if data.shape[0] != self.ndim:
            raise Exception('Expected data of dimension %d but got %d' % (self.ndim, data.shape[0]))
        
        prob = zeros(data.shape)
        for dim in xrange(self.ndim):
            prob[dim,:] = self.kde[dim].evaluate(data[dim,:])

        if perDim:
            return prob
        else:
            if abs(log10(prob.min())) * self.ndim * 2 > log10(finfo(float64).max):
                print 'WARNING: IndependentGKEDensity.pdf density returning ill-conditioned value!'
            return prod(prob, 0)


    def logpdf(self, data, perDim = False):
        '''Returns the log probability of the given samples.

        logpdf is recommended when using large dimension and perDim = False'''

        if not self.isFit():
            return Exception('Not yet fit')
        if data.shape[0] != self.ndim:
            raise Exception('Expected data of dimension %d but got %d' % (self.ndim, data.shape[0]))
        
        logprob = zeros(data.shape)
        for dim in xrange(self.ndim):
            logprob[dim,:] = log(self.kde[dim].evaluate(data[dim,:]))
        
        if perDim:
            return logprob
        else:
            return sum(logprob, 0)


def demo(resultsDir = None):
    Ndat  = 5000
    Nsamp = 5000
    data = zeros((6, Ndat))
    data[0:2,:] = random.normal(0, 1, (2,Ndat))
    data[2:4,:] = random.uniform(-3, 3, (2,Ndat))
    data[4:6,:] = random.laplace(0, 1, (2,Ndat))

    pyplot.figure()
    for ii in range(6):
        pyplot.subplot(3,2,ii+1)
        pyplot.hist(data[ii,:], bins=30, normed=True)
        if ii == 0: pyplot.title('data')
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'dimensions_data.png'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'dimensions_data.pdf'))
    
    density = IndependentGKEDensity()
    density.fit(data)

    sample = density.sample(Nsamp)
    pyplot.figure()
    for ii in range(6):
        pyplot.subplot(3,2,ii+1)
        pyplot.hist(sample[ii,:], bins=30, normed=True)
        if ii == 0: pyplot.title('sampled')
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'dimensions_sampled.png'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'dimensions_sampled.pdf'))

    dataPdf = density.pdf(data)
    dataLogPdf = density.logpdf(data)
    samplePdf = density.pdf(sample)
    sampleLogPdf = density.logpdf(sample)

    print 'dataPdf =', dataPdf
    print 'dataLogPdf = ', dataLogPdf
    print 'samplePdf =', samplePdf
    print 'sampleLogPdf = ', sampleLogPdf

    pyplot.figure()
    pyplot.hist(dataPdf,   bins=30, normed=True, color='b')
    pyplot.hist(samplePdf, bins=30, normed=True, color='r')
    pyplot.legend(('data pr.', 'sample pr.'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'prob.png'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'prob.pdf'))

    pyplot.figure()
    pyplot.hist(dataLogPdf,   bins=30, normed=True, color='b')
    pyplot.hist(sampleLogPdf, bins=30, normed=True, color='r')
    pyplot.legend(('data log pr.', 'sample log pr.'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'log_prob.png'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, 'log_prob.pdf'))

    if not resultsDir:
        pyplot.show()



if __name__ == '__main__':
    resman.start('junk', diary = False)
    demo(resman.rundir)
    resman.stop()
