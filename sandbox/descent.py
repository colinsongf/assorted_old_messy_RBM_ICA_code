#! /usr/bin/env ipythonpl

from pylab import *
from numpy import *

from numdifftools import Gradient



dim = 1000
frng = random.RandomState(0)
factors = frng.normal(0, 4, dim)



def costFn(xx):
    '''Global minimum is cost 0 at the origin.'''
    lambd = .1
    epsilon = .01
    #cost = prod(-cos(xx * factors) + 1) + norm(xx)
    sumsq = (xx**2).sum() + epsilon

    cost = (-cos(xx * factors) + 1).sum() + lambd * (sqrt(sumsq) - sqrt(epsilon))

    grad = sin(xx * factors)*factors + (lambd/sqrt(sumsq)) * xx
    
    return cost, grad



def testCostGrads():
    random.seed(0)

    for ii in range(10):
        xx = random.normal(0, 1, dim) * ii      # so first point is origin
        cost, sGrad = costFn(xx)
        print 'Current cost is:', cost, 
        #print 'Gradient (symbolic) is:', sGrad

        nGradFn = Gradient(lambda xx: costFn(xx)[0])
        nGrad = nGradFn(xx)
        #print 'Gradient (finite diff) is:', nGrad
        #print 'diff is', sGrad - nGrad
        print 'norm(diff) is', norm(sGrad - nGrad)
        print '   grad err / est err = ', abs(sGrad - nGrad) / nGradFn.error_estimate



def descend(adadelta = False, lr = None, decayRho = None, epsilon = None, color = None, dashes = None):
    rng = random.RandomState(0)
    x0 = rng.normal(0, 1, dim)
    print 'cost for x0 is', costFn(x0)[0]
    print 'cost for x* is', costFn(x0 * 0)[0]

    maxiter = 1000
    costs = zeros(maxiter)
    xcur = x0

    if adadelta:
        if decayRho is None: decayRho = .95
        if epsilon is None: epsilon = 1e-6
        expG2 = 0
        expDx2 = 0
        expG2s = []
        expDx2s = []
    else:
        if lr is None: lr = .01

    for ii in xrange(maxiter):
        cost, grad = costFn(xcur)
        costs[ii] = cost
        if adadelta:
            expG2 = decayRho * expG2 + (1-decayRho) * grad**2
            deltaX = -sqrt(expDx2 + epsilon)/sqrt(expG2 + epsilon) * grad
            expDx2 = decayRho * expDx2 + (1-decayRho) * deltaX**2
            xcur += deltaX
            expG2s.append(expG2)
            expDx2s.append(expDx2)
        else:
            xcur = xcur - lr * grad
        if ii % 100 == 0:
            print ' %3d cost: %s' % (ii, cost)
    #print 'final x:', xcur
    #print 'final x:', xcur / pi

    #plot(costs)
    if color is None:
        h1, = semilogy(costs)
    else:
        print 'called with', color
        h1, = semilogy(costs, color = color)
    if dashes:
        h1.set_dashes(dashes)
    if adadelta:
        #plot(expG2s)
        #plot(expDx2s)
        pass



def main():
    clf()
    descend(adadelta = False, lr = 1e-4, color = 'g', dashes = [2,2])
    descend(adadelta = False, lr = 1e-3, color = 'g', dashes = [4,4])
    descend(adadelta = False, lr = 1e-2, color = 'g', dashes = [6,6])
    descend(adadelta = False, lr = 1e-1, color = 'g', dashes = [8,8])
    descend(adadelta = True, epsilon = 1e-8, color = 'b', dashes = [2,2])
    descend(adadelta = True, epsilon = 1e-6, color = 'b', dashes = [4,4])
    descend(adadelta = True, epsilon = 1e-4, color = 'b', dashes = [6,6])
    descend(adadelta = True, epsilon = 1e-2, color = 'b', dashes = [8,8])
    legend(('GD 1-e4', 'GD 1-e3', 'GD 1-e2', 'GD 1-e1', 'AD 1e-8', 'AD 1e-6', 'AD 1e-4', 'AD 1e-2'))
    


if __name__ == '__main__':
    main()
