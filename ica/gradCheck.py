#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

from numpy import *
from scipy.linalg import norm
from IPython import embed
from numdifftools import Derivative, Gradient
import pdb



############################
#
#  Checking functions
#
############################



def numericalCheckVectorGrad(costGradFunction, xx, otherArgs = None):
    '''For costGradFunction of the form:

    cost, grad = costGradFunction(xx, *otherArgs)

    where xx and grad are vectors of the same shape, this function
    checks that grad matches a numerically approxmiated grad from
    the cost term.
    '''

    if otherArgs is None:
        cost,anaGrad = costGradFunction(xx)
        gradFn = Gradient(lambda x: costGradFunction(x)[0])
    else:
        cost,anaGrad = costGradFunction(xx, *otherArgs)
        gradFn = Gradient(lambda x: costGradFunction(x, *otherArgs)[0])
    numGrad = gradFn(xx)

    #if any(abs(anaGrad - numGrad) > 10 * gradFn.error_estimate) or gradFn.error_estimate.max() > 1e-10:
    if abs(anaGrad - numGrad).max() > 1e-8:
        print '[ERROR] %s: Possible failure!' % costGradFunction.__name__

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad) / gradFn.error_estimate
    else:
        print '[OK]    %s: Vector analytical gradient matches numerical approximation (max diff: %s).' % (costGradFunction.__name__, abs(anaGrad - numGrad).max())



def numericalCheckMatrixGrad(costGradFunction, XX, otherArgs):
    '''For costGradFunction of the form:

    cost, grad = costGradFunction(XX, *otherArgs)

    where XX and grad are matrices of the same shape, this function
    checks that grad matches a numerically approxmiated grad from
    the cost term.
    '''

    cost,anaGrad = costGradFunction(XX, *otherArgs)

    def unflatteningWrapper(func, xflat, xshape, *args):
        return func(reshape(xflat, xshape), *args)

    gradFn = Gradient(lambda xflat: unflatteningWrapper(costGradFunction, xflat, XX.shape, *otherArgs)[0])
    numGradFlat = gradFn(XX.flatten())
    numGrad = reshape(numGradFlat, XX.shape)

    #if any(abs(anaGrad - numGrad).flatten() > 10 * gradFn.error_estimate) or gradFn.error_estimate.max() > 1e-10:
    if abs(anaGrad - numGrad).max() > 1e-8:
        print '[ERROR] %s: Possible failure!' % costGradFunction.__name__

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad).flatten() / gradFn.error_estimate
    else:
        print '[OK]    %s: Matrix analytical gradient matches numerical approximation (max diff: %s).' % (costGradFunction.__name__, abs(anaGrad - numGrad).max())



def checkVectorMatrixGradientsEqual(costGradFunction_vec, costGradFunction_mat, XX, otherArgs):
    '''For costGradFunction_vec of the form:

    vCost, vGrad = costGradFunction_vec(xx, *otherArgs)

    where xx and grad are vectors of the same shape, and

    mCost, mGrad = costGradFunction_mat(XX, *otherArgs)

    where XX and grad are matrices of the same shape, this function checks
    that each column jj of mGrad is equal to vGrad for xx = XX[:,jj] and that
    mCost = the total of vCost over all columns.
    '''

    mCost,mGrad = costGradFunction_mat(XX, *otherArgs)

    nColumns = XX.shape[1]

    vCostTotal = 0
    failure = False
    for colIdx in range(nColumns):
        vCost, vGrad = costGradFunction_vec(XX[:,colIdx], *otherArgs)
        vCostTotal += vCost
        if abs(vGrad - mGrad[:,colIdx]).max() > 1e-10:
            if not failure:
                print '[ERROR] %s vs %s: Possible failure with column %d!' % (costGradFunction_vec.__name__, costGradFunction_mat.__name__, colIdx)

                print 'cost:', cost
                print '\ncol grad:', vgrad
                print '\ncol of mat grad:', anaGrad[:,colIdx]
                print '\ndifference:', vgrad - anaGrad[:,colIdx]
                print '\ndifference.max():', abs(vgrad - anaGrad[:,colIdx]).max

                print '(further errors supressed)'
            failure = True
    if not failure:
        print '[OK]    %s vs %s: columns of matrix gradient match vector version.' % (costGradFunction_vec.__name__, costGradFunction_mat.__name__)
    if abs(mCost - vCostTotal) > nColumns * 1e-12:
        print '[ERROR] %s vs %s: Costs do not match: %s != %s' % (costGradFunction_vec.__name__, costGradFunction_mat.__name__, mCost, vCostTotal)

