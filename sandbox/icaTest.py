#! /usr/bin/env ipython --pylab

from numpy import *
import pdb
from matplotlib import pyplot
from scipy.optimize import fmin_bfgs

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
    tt = array(theta).flatten()[0]  # handle numbers or arrays passed as arguments
    return array([[cos(tt), -sin(tt)],
                  [sin(tt), cos(tt)]])



def main():
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

    minfnK    = lambda theta: -kurt(dot(rotMat(theta)[0,:], W.T).T)**2
    minfnNEnt = lambda theta: -negentropy(dot(rotMat(theta)[0,:], W.T).T)
    minfnLC10 = lambda theta: -logcosh10(dot(rotMat(theta)[0,:], W.T).T)
    minfnLC15 = lambda theta: -logcosh15(dot(rotMat(theta)[0,:], W.T).T)
    minfnLC20 = lambda theta: -logcosh20(dot(rotMat(theta)[0,:], W.T).T)
    minfnNExp = lambda theta: -negexp(dot(rotMat(theta)[0,:], W.T).T)

    # Change this line to change which objective is used
    minfn = minfnK

    angle0 = 0
    xopt = fmin_bfgs(minfn, angle0)

    mnval = array([minfn(aa) for aa in thetas])

    pyplot.figure(4)
    pyplot.title('objective vs. angle')
    #pyplot.plot(thetas, kurt1, 'bo-', thetas, mnval, 'k', xopt, minfn(xopt), 'ko')
    pyplot.plot(thetas, mnval, 'b', xopt, minfn(xopt), 'ko')

    pyplot.figure(5)
    pyplot.title('different gaussianness measures vs. angle')
    pyplot.subplot(6,1,1);  pyplot.plot(thetas, array([minfnK(aa) for aa in thetas]))
    pyplot.subplot(6,1,2);  pyplot.plot(thetas, array([minfnNEnt(aa) for aa in thetas]))
    pyplot.subplot(6,1,3);  pyplot.plot(thetas, array([minfnLC10(aa) for aa in thetas]))
    pyplot.subplot(6,1,4);  pyplot.plot(thetas, array([minfnLC15(aa) for aa in thetas]))
    pyplot.subplot(6,1,5);  pyplot.plot(thetas, array([minfnLC20(aa) for aa in thetas]))
    pyplot.subplot(6,1,6);  pyplot.plot(thetas, array([minfnNExp(aa) for aa in thetas]))

    Ropt = rotMat(xopt)
    Recon = dot(W, Ropt.T)
    print 'kurt(r1) =', kurt(Recon[:,0])
    print 'kurt(r2) =', kurt(Recon[:,1])

    print
    print 'objective(s1) =', minfn(s1)
    print 'objective(s2) =', minfn(s2)
    print 'objective(w1) =', minfn(w1)
    print 'objective(w2) =', minfn(w2)
    print 'objective(r1) =', minfn(Recon[:,0])
    print 'objective(r2) =', minfn(Recon[:,1])

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
    import ipdb; ipdb.set_trace()
    


if __name__ == '__main__':
    main()
    #pyplot.show()
    print 'shown'
    
    #import IPython; IPython.embed()
    #import IPython.core
    #IPython.core.ipapi.launch_new_instance(locals())
    #IPython.core.interactiveshell.InteractiveShell()

    #from IPython.core.debugger import Tracer
    #debug = Tracer()
    #debug()


    
