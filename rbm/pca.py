#! /usr/bin/env python

from numpy import array, dot, random, linalg, sqrt, asarray



class PCA:
    def __init__(self, xx):
        '''
        Inspired by PCA in matplotlib.mlab

        Compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

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

        self.centeredXX  = self.center(xx)

        UU, ss, self.Vh = linalg.svd(self.centeredXX, full_matrices=False)

        self.var = ss**2 / float(self.nn)
        self.std = sqrt(self.var)
        self.fracVar = self.var / self.var.sum()
        self.fracStd = self.std / self.std.sum()


    def toPC(self, xx, numDims = None):
        '''Center the xx and project it onto the principle components'''

        xx = asarray(xx)

        if xx.shape[-1] != self.mm:
            raise ValueError('Expected an array with dims[-1] == %d' % self.mm)

        if numDims is None:
            numDims = xx.shape[-1]
        
        centered = self.center(xx)
        pc = dot(centered, self.Vh[0:numDims,:].T) / self.std[0:numDims]   # more efficient
        
        return pc


    def fromPC(self, pc):
        '''Project the given principle components back to the original space and uncenter them.'''

        pc = asarray(pc)

        ndims = pc.shape[1]
        
        centered = dot(pc * self.std[0:ndims], self.Vh[0:ndims,:])
        xx = self.uncenter(centered)

        return xx


    def center(self, xx):
        '''Center the data using the mean and sigma from the training set'''

        return (xx - self.mu) / self.sigma


    def uncenter(self, cc):
        '''Undo the operation of center'''

        return (cc * self.sigma) + self.mu



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

    pca = PCA(data1)

    print 'data1 centered\n',       pca.center(data1)
    print 'data1 uncentered\n',     pca.uncenter(pca.center(data1))
    print 'data1 toPC\n',           pca.toPC(data1)
    print 'data1 fromPC\n',         pca.fromPC(pca.toPC(data1))
    print 'data1 toPC (1 dim)\n',   pca.toPC(data1, 1)
    print 'data1 fromPC (1 dim)\n', pca.fromPC(pca.toPC(data1, 1))

    pc1 = pca.toPC(data1)
    recon1 = pca.fromPC(pc1)

    pc2 = pca.toPC(data2)
    recon2 = pca.fromPC(pc2)

    pc2_1dim = pca.toPC(data2, 1)
    recon2_1dim = pca.fromPC(pc2_1dim)
    
    pyplot.figure()

    pyplot.subplot(3,3,1)
    pyplot.plot(data1[:,0], data1[:,1], 'o')
    pyplot.title('data1')

    pyplot.subplot(3,3,2)
    pyplot.plot(pc1[:,0], pc1[:,1], 'o')
    pyplot.title('pc1')

    pyplot.subplot(3,3,3)
    pyplot.plot(recon1[:,0], recon1[:,1], 'o')
    pyplot.title('recon1')

    pyplot.subplot(3,3,4)
    pyplot.plot(data2[:,0], data2[:,1], 'o')
    pyplot.title('data2')

    pyplot.subplot(3,3,5)
    pyplot.plot(pc2[:,0], pc2[:,1], 'o')
    pyplot.title('pc2')

    pyplot.subplot(3,3,6)
    pyplot.plot(recon2[:,0], recon2[:,1], 'o')
    pyplot.title('recon2')

    pyplot.subplot(3,3,7)
    pyplot.plot(data2[:,0], data2[:,1], 'o')
    pyplot.title('data2')

    pyplot.subplot(3,3,8)
    pyplot.plot(pc2_1dim[:,0], pc2_1dim[:,0] * 0, 'o')
    pyplot.title('pc2[1 dim]')

    pyplot.subplot(3,3,9)
    pyplot.plot(recon2_1dim[:,0], recon2_1dim[:,1], 'o')
    pyplot.title('recon2[1 dim]')

    pyplot.show()



if __name__ == '__main__':
    testPca()
