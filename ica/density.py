#! /usr/bin/env ipythonpl

import pdb
import os
import time
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


    def pdf(self, data, perDim = False, quiet = False):
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
            prodprob = prod(prob, 0)
            ppmin = prodprob.min()
            ppmax = prodprob.max()
            if (ppmin == 0 or   # check first or risk log10(ppmin) error
                abs(log10(ppmin)) > .9 * log10(finfo(float64).max) or
                abs(log10(ppmax)) > .9 * log10(finfo(float64).max)):
                if not quiet:
                    print 'WARNING: IndependentGKEDensity.pdf density returning ill-conditioned value!'
            return prodprob


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


def demo(resultsDir = None, Ndat = 500, Nsamp = 500, Ndim = 6, scalingTest = False):
    if Ndim == 6:
        data = zeros((Ndim, Ndat))
        data[0:2,:] = random.normal(0, 1, (2,Ndat))
        data[2:4,:] = random.uniform(-3, 3, (2,Ndat))
        data[4:6,:] = random.laplace(0, 1, (2,Ndat))
    else:
        data = random.laplace(0, 1, (Ndim, Ndat))

    pyplot.figure()
    for ii in range(6):
        pyplot.subplot(3,2,ii+1)
        pyplot.hist(data[ii,:], bins=30, normed=True)
        if ii == 0: pyplot.title('data')
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_dimensions_data.png' % (Ndat,Nsamp,Ndim)))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_dimensions_data.pdf' % (Ndat,Nsamp,Ndim)))
    
    density = IndependentGKEDensity()
    density.fit(data)

    t0 = time.time()
    sample = density.sample(Nsamp)
    timeSample = time.time() - t0
    
    pyplot.figure()
    for ii in range(6):
        pyplot.subplot(3,2,ii+1)
        pyplot.hist(sample[ii,:], bins=30, normed=True)
        if ii == 0: pyplot.title('sampled')
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_dimensions_sampled.png' % (Ndat,Nsamp,Ndim)))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_dimensions_sampled.pdf' % (Ndat,Nsamp,Ndim)))

    dataPdf = density.pdf(data, quiet = scalingTest)
    dataLogPdf = density.logpdf(data)
    samplePdf = density.pdf(sample, quiet = scalingTest)
    t0 = time.time()
    sampleLogPdf = density.logpdf(sample)
    timePdf = time.time() - t0

    if not scalingTest:
        print 'dataPdf =', dataPdf[:10], '...'
        print 'dataLogPdf = ', dataLogPdf[:10], '...'
        print 'samplePdf =', samplePdf[:10], '...'
        print 'sampleLogPdf = ', sampleLogPdf[:10], '...'

    pyplot.figure()
    pyplot.hist(dataPdf,   bins=30, normed=True, color='b')
    pyplot.hist(samplePdf, bins=30, normed=True, color='r')
    pyplot.legend(('data pr.', 'sample pr.'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_prob.png' % (Ndat,Nsamp,Ndim)))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_prob.pdf' % (Ndat,Nsamp,Ndim)))

    pyplot.figure()
    pyplot.hist(dataLogPdf,   bins=30, normed=True, color='b')
    pyplot.hist(sampleLogPdf, bins=30, normed=True, color='r')
    pyplot.legend(('data log pr.', 'sample log pr.'))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_log_prob.png' % (Ndat,Nsamp,Ndim)))
    if resultsDir: pyplot.savefig(os.path.join(resultsDir, '%d_%d_%d_log_prob.pdf' % (Ndat,Nsamp,Ndim)))

    if not resultsDir:
        pyplot.show()

    return timeSample, timePdf



if __name__ == '__main__':
    resman.start('junk', diary = False)

    fullScalingTest = False
    if fullScalingTest:
        print 'Doing full scaling test'
        print '%6s %6s %6s %17s %17s' % ('Ndat', 'Nsamp', 'Ndim', 'timeSample', 'timePdf')
        for ii, Ndat in enumerate([100, 400]):
            for jj, Nsamp in enumerate([100, 400]):
                for kk, Ndim in enumerate([100, 400]):
                    timeSample, timePdf = demo(resman.rundir, Ndat = Ndat, Nsamp = Nsamp, Ndim = Ndim, scalingTest = True)
                    if (ii,jj,kk) == (0,0,0):
                        timeSample0 = timeSample
                        timePdf0 = timePdf
                    tupl = (Ndat, Nsamp, Ndim, timeSample, timeSample/timeSample0, timePdf, timePdf/timePdf0)
                    print '%6d %6d %6d %10f (%.2f) %10f (%.2f)' % tupl
    else:
        # just do a simple demo
        demo(resman.rundir)
    resman.stop()
