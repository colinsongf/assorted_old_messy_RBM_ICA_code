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

from cost import *



############################
#
#  Checking functions
#
############################



def numericalCheckVectorGrad(costGradFunction, xx, otherArgs):
    '''For costGradFunction of the form:

    cost, grad = costGradFunction(xx, *otherArgs)

    where xx and grad are vectors of the same shape, this function
    checks that grad matches a numerically approxmiated grad from
    the cost term.
    '''

    cost,anaGrad = costGradFunction(xx, *otherArgs)
    gradFn = Gradient(lambda x: costGradFunction(x, *otherArgs)[0])
    numGrad = gradFn(xx)

    #if any(abs(anaGrad - numGrad) > 10 * gradFn.error_estimate) or gradFn.error_estimate.max() > 1e-10:
    if abs(anaGrad - numGrad).max() > 1e-8:
        print '%s: Possible failure!' % costGradFunction.__name__

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad) / gradFn.error_estimate
    else:
        print '%s: Vector analytical gradient matches numerical approximation! (max diff: %s)' % (costGradFunction.__name__, abs(anaGrad - numGrad).max())



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
        print '%s: Possible failure!' % costGradFunction.__name__

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad).flatten() / gradFn.error_estimate
    else:
        print '%s: Matrix analytical gradient matches numerical approximation! (max diff: %s)' % (costGradFunction.__name__, abs(anaGrad - numGrad).max())



############################
#
#  Actual tests
#
############################



def check_negAvgTwoNorm_vec():
    random.seed(0)
    dim = 100
    NN  = 1000
    XX = random.normal(0, 1, (dim,NN))
    vv = random.normal(0, 1, (dim,))
    
    numericalCheckVectorGrad(negAvgTwoNorm_vec, vv, (XX,))
    
    

def check_negAvgTwoNorm_mat():
    random.seed(0)
    dim = 10
    NN  = 100
    kk = 40
    XX = random.normal(0, 1, (dim,NN))
    VV = random.normal(0, 1, (dim,kk))

    numericalCheckMatrixGrad(negAvgTwoNorm_mat, VV, (XX,))



def check_vs_vec_negAvgTwoNorm_mat():
    random.seed(0)
    dim = 100
    NN  = 1000
    kk = 400
    XX = random.normal(0, 1, (dim,NN))
    VV = random.normal(0, 1, (dim,kk))

    cost,anaGrad = negAvgTwoNorm_mat(VV, XX)

    vcostTotal = 0
    failure = False
    for colIdx in range(kk):
        vcost, vgrad = negAvgTwoNorm_vec(VV[:,colIdx], XX)
        vcostTotal += vcost
        if not all(abs(vgrad - anaGrad[:,colIdx]) < 1e-10):
            if not failure:
                print 'Possible failure with column %d!' % colIdx

                print 'cost:', cost
                print '\ncol grad:', vgrad
                print '\ncol of mat grad:', anaGrad[:,colIdx]
                print '\ndifference:', vgrad - anaGrad[:,colIdx]
                print '\ndifference.max():', abs(vgrad - anaGrad[:,colIdx]).max

                print '(further errors supressed)'
            failure = True
    if not failure:
        print 'Gradient tests passed!'
    if abs(cost - vcostTotal) > kk * 1e-12:
        print 'Costs do not match: %s != %s' % (cost, vcostTotal)

    #embed()



def main():
    #check_negAvgTwoNorm_vec()
    #check_negAvgTwoNorm_mat()
    #check_vs_vec_negAvgTwoNorm_mat()

    check_negAvgTwoNorm_mat2()

if __name__ == '__main__':
    main()
