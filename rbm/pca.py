#! /usr/bin/env python

import pdb
from numpy import array, dot, random, linalg, sqrt, asarray, cov



class PCA_SVD:
    def __init__(self, xx):
        '''
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



class PCA:
    def __init__(self, xx):
        '''
        Inspired by PCA in matplotlib.mlab

        Compute the principle components of a dataset and stores the
        mean, sigma, and SVD of sigma for the data.  Use toPC and
        fromPC to project the data onto a reduced set of dimensions
        and back. This version takes the SVD of the covariance matrix.

        Inputs:

          *xx*: a numobservations x numdims array

        Attrs:

          *nn*, *mm*: the dimensions of xx

          *mu* : a numdims array of means of xx

          *sigma* : ?????????

          *var* : the average amount of variance of each of the principal components

          *std* : sqrt of var

          *fracVar* : the fractional amount of variance from each principal component

          *fracStd* : sqrt of fracVar
        '''

        self.nn, self.mm = xx.shape
        if self.nn < self.mm:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.mu          = xx.mean(axis=0)
        self.centeredXX  = self.center(xx)
        self.sigma       = dot(self.centeredXX.T, self.centeredXX) / self.nn

        # UU and VV are transpose of each other
        self.UU, self.ss, VV = linalg.svd(self.sigma, full_matrices = False)

        self.var = self.ss / float(self.nn)
        self.std = sqrt(self.var)
        self.fracVar = self.var / self.var.sum()
        self.fracStd = self.std / self.std.sum()


    def pc(self):
        return self.UU


    def toWhitePC(self, xx, numDims = None, epsilon = 0):
        return self.toPC(xx, numDims = numDims, whiten = True, epsilon = epsilon)
    

    def toPC(self, xx, numDims = None, whiten = False, epsilon = 0):
        '''Center the xx and project it onto the principle components.

        Called \tilde{x} on UFLDL wiki page.'''

        xx = asarray(xx)

        if xx.shape[-1] != self.mm:
            raise ValueError('Expected an array with dims[-1] == %d' % self.mm)

        if not whiten and epsilon != 0:
            raise Exception('Probable misuse: epsilon != 0 but whitening is off.')

        if numDims is None:
            numDims = xx.shape[-1]

        centered = self.center(xx)
        #pc = dot(centered, self.UU[:,0:numDims]) / self.std[0:numDims]

        if whiten:
            pc = dot(centered, self.UU[:,0:numDims]) / sqrt(self.ss[0:numDims] + epsilon)
        else:
            pc = dot(centered, self.UU[0:numDims,:].T)

        #pdb.set_trace()

        return pc


    def fromWhitePC(self, pc):
        return self.fromPC(pc, whiten = True)


    def fromPC(self, pc, whiten = False):
        '''Project the given principle components back to the original
        space and uncenter them.'''

        pc = asarray(pc)

        numDims = pc.shape[1]

        if whiten:
            # actually the same either way :)        HERE provide unwhiten version.
            centered = dot(pc, self.UU[:,0:numDims].T)
        else:
            centered = dot(pc, self.UU[:,0:numDims].T)

        xx = self.uncenter(centered)

        return xx


    def zca(self, xx, numDims, epsilon = 0):
        '''Return Zero-phase whitening filter version.'''

        pc = self.toWhitePC(xx, numDims = numDims, epsilon = epsilon)
        return self.fromWhitePC(pc)


    def center(self, xx):
        '''Center the data using the mean from the training set. Does
        not ensure that each dimension has std = 1.'''

        return xx - self.mu


    def uncenter(self, cc):
        '''Undo the operation of center'''

        return cc + self.mu




def testPca():
    from matplotlib import pyplot
    random.seed(1)
    
    NN = 10
    transform = array([[2, 3.5], [3.5, 8]])
    #transform = array([[2, 4.5], [2.5, 8]])
    data1 = random.randn(NN,2)
    data1 = dot(data1, transform)
    data1[:,0] += 4
    data1[:,1] += -2
    data2 = random.randn(NN,2)
    data2 = dot(data2, transform)
    data2[:,0] += 4
    data2[:,1] += -2

    print 'data1\n', data1
    print 'data2\n', data2
    print

    print 'PCA_SVD'
    testPcaHelper(data1, data2, usePcaSvd = True)
    print '\nPCA'
    testPcaHelper(data1, data2, usePcaSvd = False)

    pyplot.show()



def testPcaHelper(data1, data2, usePcaSvd = False):
    '''Helper function for testing either PCA_SVD or PCA.'''

    from matplotlib import pyplot

    if usePcaSvd:
        pca = PCA_SVD(data1)
    else:
        pca = PCA(data1)

    print 'Principle components (columns)\n', pca.pc()
    print 'data1 centered\n',       pca.center(data1)
    print 'data1 uncentered\n',     pca.uncenter(pca.center(data1))
    print 'data1 toPC\n',           pca.toPC(data1)
    print 'data1 cov(toPC.T)\n',    cov(pca.toPC(data1).T, bias = 1)
    if not usePcaSvd:
        print 'data1 cov(toWhitePC.T)\n',    cov(pca.toWhitePC(data1).T, bias = 1)
    print 'data1 fromPC\n',         pca.fromPC(pca.toPC(data1))
    print 'data1 toPC (1 dim)\n',   pca.toPC(data1, 1)
    print 'data1 fromPC (1 dim)\n', pca.fromPC(pca.toPC(data1, 1))
    if not usePcaSvd:
        print 'data1 zca\n',    pca.zca(data1, 1)

    pc1 = pca.toPC(data1)
    if not usePcaSvd:
        pc1white = pca.toWhitePC(data1)
    recon1 = pca.fromPC(pc1)

    pc2 = pca.toPC(data2)
    if not usePcaSvd:
        pc2white = pca.toWhitePC(data1)
    recon2 = pca.fromPC(pc2)

    pc2_1dim = pca.toPC(data2, 1)
    recon2_1dim = pca.fromPC(pc2_1dim)

    print 'fromPC(toPC(data2, 1))\n', recon2_1dim

    print 'reconstruction error:', ((recon2_1dim - data2)**2).sum()

    pyplot.figure()

    pyplot.subplot(3,4,1)
    pyplot.plot(data1[:,0], data1[:,1], 'o')
    str = 'PCA_SVD' if usePcaSvd else 'PCA'
    pyplot.title(str + ': data1')

    pyplot.subplot(3,4,2)
    pyplot.plot(pc1[:,0], pc1[:,1], 'o')
    pyplot.title('pc1')

    if not usePcaSvd:
        pyplot.subplot(3,4,3)
        pyplot.plot(pc1white[:,0], pc1white[:,1], 'o')
        pyplot.title('pc1white')

    pyplot.subplot(3,4,4)
    pyplot.plot(recon1[:,0], recon1[:,1], 'o')
    pyplot.title('recon1')

    pyplot.subplot(3,4,5)
    pyplot.plot(data2[:,0], data2[:,1], 'o')
    pyplot.title('data2')

    pyplot.subplot(3,4,6)
    pyplot.plot(pc2[:,0], pc2[:,1], 'o')
    pyplot.title('pc2')

    if not usePcaSvd:
        pyplot.subplot(3,4,7)
        pyplot.plot(pc2white[:,0], pc2white[:,1], 'o')
        pyplot.title('pc2white')

    pyplot.subplot(3,4,8)
    pyplot.plot(recon2[:,0], recon2[:,1], 'o')
    pyplot.title('recon2')

    pyplot.subplot(3,4,9)
    pyplot.plot(data2[:,0], data2[:,1], 'o')
    pyplot.title('data2')

    pyplot.subplot(3,4,10)
    pyplot.plot(pc2_1dim[:,0], pc2_1dim[:,0] * 0, 'o')
    pyplot.title('pc2[1 dim]')

    pyplot.subplot(3,4,11)
    pyplot.semilogy(pca.var, 'o-')
    pyplot.title('pc1.var')

    pyplot.subplot(3,4,12)
    pyplot.plot(recon2_1dim[:,0], recon2_1dim[:,1], 'o')
    pyplot.title('recon2[1 dim]')



if __name__ == '__main__':
    testPca()
