#! /usr/bin/env ipythonwx

from numpy import *
from matplotlib import pyplot
from scipy.optimize import fmin_bfgs

from rbm.pca import PCA



def kurt(xx):
    # HERE
    return mean(xx**4, axis=0) - 3 * mean(xx**2, axis=0)**2



def rotMat(theta):
    return array([[cos(theta), -sin(theta)],
                  [sin(theta), cos(theta)]])



def main():
    N = 200
    tt = linspace(0, 10, N)

    # make sources
    s1 = 4 + cos(tt)
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
    
    X = dot(S, A)
    #print 'X', X

    x1 = X[:,0]
    x2 = X[:,1]

    print 'kurt(x1) =', kurt(x1)
    print 'kurt(x2) =', kurt(x2)

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

    W = pca.toWhitePC(X)

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
    kurt2 = 0 * thetas
    for ii, theta in enumerate(thetas):
        kurt1[ii] = kurt(dot(W, rotMat(theta))[:,0])
        kurt2[ii] = kurt(dot(W, rotMat(theta))[:,1])

    angle0 = 0
    #xopt = fmin_bfgs(lambda aa : -sum(kurt(dot(W, rotMat(aa))) ** 2),
    #                 angle0)
    xopt = fmin_bfgs(lambda aa : -max(kurt(dot(W, rotMat(aa))) ** 2),
                     angle0)

    mnval = array([-max(kurt(dot(W, rotMat(aa))) ** 2) for aa in thetas])

    pyplot.figure(4)
    pyplot.title('kurtosis vs. angle')
    pyplot.plot(thetas, kurt1, 'bo-', thetas, kurt2, 'ro-', thetas, kurt1 + kurt2, 'go-', thetas, mnval, 'k')
    pyplot.hold(True)
    pyplot.plot(xopt, -max(kurt(dot(W, rotMat(xopt))) ** 2), 'ko')
    pyplot.plot(thetas, array([-max(kurt(dot(W, rotMat(aa))) ** 2) for aa in thetas]), 'co-')

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


    
