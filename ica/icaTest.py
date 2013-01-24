#! /usr/bin/env ipythonpl

import pdb
import os, sys
from numpy import *
from matplotlib import pyplot
from scipy.optimize import fmin_bfgs

from GitResultsManager import resman
from rbm.pca import PCA



def assertMean0Var1(xx):
    '''Checks that the data is mean 0, variance 1'''
    assert(abs(mean(xx)) < 1e-10)
    assert(abs(var(xx) - 1) < 1e-10)



def kurt(xx):
    assertMean0Var1(xx)
    if len(xx.shape) > 1:
        pdb.set_trace()
    return mean(xx**4, axis=0) - 3 * mean(xx**2, axis=0)**2



def negentropy(xx):
    '''Assuming xx is zero mean and unit variance.'''

    assertMean0Var1(xx)
    return mean(xx**3)**2 / 12.0 + kurt(xx)**2 / 48.0



def logcosh10(xx):
    '''Assuming xx is zero mean and unit variance.'''
    assertMean0Var1(xx)
    a = 1.0
    return (mean(1/a * log(cosh(a*xx))) - 0.37457) ** 2
def logcosh15(xx):
    '''Assuming xx is zero mean and unit variance.'''
    assertMean0Var1(xx)
    a = 1.5
    return (mean(1/a * log(cosh(a*xx))) - 0.46729) ** 2
def logcosh20(xx):
    '''Assuming xx is zero mean and unit variance.'''
    assertMean0Var1(xx)
    a = 2.0
    return (mean(1/a * log(cosh(a*xx))) - 0.52833) ** 2



def negexp(xx):
    '''Assuming xx is zero mean and unit variance.'''

    assertMean0Var1(xx)
    return (mean(-exp(-xx**2/2.0)) - -1.0/sqrt(2)) ** 2



def rotMat(theta):
    # handle numbers or arrays passed as arguments
    tt = array(theta).flatten()
    if len(tt) > 1:
        raise Exception('got long theta, probably an error')
    tt = tt[0]
    return array([[cos(tt), -sin(tt)],
                  [sin(tt), cos(tt)]])



def main(addNoise = 0, savedir = None, doFastICA = False):
    N = 200
    tt = linspace(0, 10, N)

    # make sources
    s1 = 4 + cos(tt*5)
    s2 = tt % 2

    s1 -= mean(s1)
    s1 /= std(s1)
    s2 -= mean(s2)
    s2 /= std(s2)

    pyplot.figure(1)
    pyplot.subplot(4,1,1)
    pyplot.title('original sources')
    pyplot.plot(tt, s1, 'bo-')
    pyplot.subplot(4,1,2)
    pyplot.plot(tt, s2, 'bo-')

    A = array([[3, 1], [-2, .3]])

    S = vstack((s1, s2)).T
    #print 'S', S
    print 'kurt(s1) =', kurt(s1)
    print 'kurt(s2) =', kurt(s2)
    print ' negentropy(s1) =', negentropy(s1)
    print ' negentropy(s2) =', negentropy(s2)
    print ' logcosh10(s1) =', logcosh10(s1)
    print ' logcosh10(s2) =', logcosh10(s2)
    print ' logcosh15(s1) =', logcosh15(s1)
    print ' logcosh15(s2) =', logcosh15(s2)
    print ' logcosh20(s1) =', logcosh20(s1)
    print ' logcosh20(s2) =', logcosh20(s2)
    print ' negexp(s1) =', negexp(s1)
    print ' negexp(s2) =', negexp(s2)
    
    X = dot(S, A)

    if addNoise > 0:
        print 'Adding noise!'
        X += random.normal(0, addNoise, X.shape)
    
    #print 'X', X

    x1 = X[:,0]
    x2 = X[:,1]

    #print 'kurt(x1) =', kurt(x1)
    #print 'kurt(x2) =', kurt(x2)

    pyplot.subplot(4,1,3)
    pyplot.title('observed signal')
    pyplot.plot(tt, x1, 'ro-')
    pyplot.subplot(4,1,4)
    pyplot.plot(tt, x2, 'ro-')

    pyplot.figure(2)
    pyplot.subplot(4,1,1)
    pyplot.title('original sources')
    pyplot.hist(s1)
    pyplot.subplot(4,1,2)
    pyplot.hist(s2)
    pyplot.subplot(4,1,3)
    pyplot.title('observed signal')
    pyplot.hist(x1)
    pyplot.subplot(4,1,4)
    pyplot.hist(x2)

    pca = PCA(X)

    #W = pca.toWhitePC(X)
    W = pca.toZca(X)

    w1 = W[:,0]
    w2 = W[:,1]

    print 'kurt(w1) =', kurt(w1)
    print 'kurt(w2) =', kurt(w2)

    pyplot.figure(3)
    pyplot.subplot(4,2,1)
    pyplot.title('observed signal')
    pyplot.hist(x1)
    pyplot.subplot(4,2,3)
    pyplot.hist(x2)
    pyplot.subplot(2,2,2)
    pyplot.plot(x1, x2, 'bo')

    pyplot.subplot(4,2,5)
    pyplot.title('whitened observed signal')
    pyplot.hist(w1)
    pyplot.subplot(4,2,7)
    pyplot.hist(w2)
    pyplot.subplot(2,2,4)
    pyplot.plot(w1, w2, 'bo')

    # Compute kurtosis at different angles
    thetas = linspace(0, pi, 100)
    kurt1 = 0 * thetas
    for ii, theta in enumerate(thetas):
        kurt1[ii] = kurt(dot(rotMat(theta)[0,:], W.T).T)


    # functions of data
    minfnK    = lambda data: -kurt(data)**2
    minfnNEnt = lambda data: -negentropy(data)
    minfnLC10 = lambda data: -logcosh10(data)
    minfnLC15 = lambda data: -logcosh15(data)
    minfnLC20 = lambda data: -logcosh20(data)
    minfnNExp = lambda data: -negexp(data)

    # functions of the rotation angle, given W as the data
    minAngleFnK    = lambda theta: minfnK(dot(rotMat(theta)[0,:], W.T).T)
    minAngleFnNEnt = lambda theta: minfnNEnt(dot(rotMat(theta)[0,:], W.T).T)
    minAngleFnLC10 = lambda theta: minfnLC10(dot(rotMat(theta)[0,:], W.T).T)
    minAngleFnLC15 = lambda theta: minfnLC15(dot(rotMat(theta)[0,:], W.T).T)
    minAngleFnLC20 = lambda theta: minfnLC20(dot(rotMat(theta)[0,:], W.T).T)
    minAngleFnNExp = lambda theta: minfnNExp(dot(rotMat(theta)[0,:], W.T).T)

    #########
    # Chosen objective function. Change this line to change which objective is used.
    #########
    minDataFn = minfnK 

    minAngleFn = lambda theta: minDataFn(dot(rotMat(theta)[0,:], W.T).T)

    if doFastICA:
        # Use FastICA from sklearn
        #pdb.set_trace()
        from sklearn.decomposition import FastICA
        rng = random.RandomState(1)
        ica = FastICA(random_state = rng, whiten = False)
        ica.fit(W)
        Recon = ica.transform(W)  # Estimate the sources
        #S_fica /= S_fica.std(axis=0)   # (should already be done)
        Ropt = ica.get_mixing_matrix()
    else:
        # Manually fit angle using fmin_bfgs
        angle0 = 0
        xopt = fmin_bfgs(minAngleFn, angle0)
        xopt = xopt[0] % pi
        Ropt = rotMat(xopt)
        Recon = dot(W, Ropt.T)

    mnval = array([minAngleFn(aa) for aa in thetas])

    pyplot.figure(4)
    pyplot.title('objective vs. angle')
    #pyplot.plot(thetas, kurt1, 'bo-', thetas, mnval, 'k', xopt, minAngleFn(xopt), 'ko')
    pyplot.plot(thetas, mnval, 'b')
    if not doFastICA:
        pyplot.hold(True)
        pyplot.plot(xopt, minAngleFn(xopt), 'ko')

    pyplot.figure(5)
    pyplot.title('different gaussianness measures vs. angle')
    pyplot.subplot(6,1,1); pyplot.title('Kurt'); pyplot.plot(thetas, array([minAngleFnK(aa) for aa in thetas]))
    pyplot.subplot(6,1,2); pyplot.title('NegEnt'); pyplot.plot(thetas, array([minAngleFnNEnt(aa) for aa in thetas]))
    pyplot.subplot(6,1,3); pyplot.title('LogCosh10'); pyplot.plot(thetas, array([minAngleFnLC10(aa) for aa in thetas]))
    pyplot.subplot(6,1,4); pyplot.title('LogCosh15'); pyplot.plot(thetas, array([minAngleFnLC15(aa) for aa in thetas]))
    pyplot.subplot(6,1,5); pyplot.title('LogCosh20'); pyplot.plot(thetas, array([minAngleFnLC20(aa) for aa in thetas]))
    pyplot.subplot(6,1,6); pyplot.title('NegExp'); pyplot.plot(thetas, array([minAngleFnNExp(aa) for aa in thetas]))
    
    print 'kurt(r1) =', kurt(Recon[:,0])
    print 'kurt(r2) =', kurt(Recon[:,1])

    print
    print 'objective(s1) =', minDataFn(s1)
    print 'objective(s2) =', minDataFn(s2)
    print 'objective(w1) =', minDataFn(w1)
    print 'objective(w2) =', minDataFn(w2)
    print 'objective(r1) =', minDataFn(Recon[:,0])
    print 'objective(r2) =', minDataFn(Recon[:,1])
    print 'optimal theta:',
    if doFastICA:
        print '<not computed with FastICA>'
    else:
        print xopt, '(+pi/2 =', (xopt+pi/2)%pi, ')'
    print 'Optimal rotation matrix:\n', Ropt

    pyplot.figure(6)
    pyplot.subplot(4,1,1)
    pyplot.title('original sources')
    pyplot.plot(tt, s1, 'bo-')
    pyplot.subplot(4,1,2)
    pyplot.plot(tt, s2, 'bo-')
    pyplot.subplot(4,1,3)
    pyplot.title('reconstructed sources')
    pyplot.plot(tt, Recon[:,0], 'go-')
    pyplot.subplot(4,1,4)
    pyplot.plot(tt, Recon[:,1], 'go-')

    #pyplot.show()

    if savedir:
        figname = lambda ii : os.path.join(savedir, 'figure_%02d.png' % ii)
        for ii in range(6):
            pyplot.figure(ii+1)
            pyplot.savefig(figname(ii+1))
        print 'plots saved in', savedir
    else:
        import ipdb; ipdb.set_trace()
    


if __name__ == '__main__':
    resman.start('junk', diary = False)
    main(addNoise = 0,
         #savedir = resman.rundir,     # comment out to show plots instead of saving
         doFastICA = False,
         )
    resman.stop()
