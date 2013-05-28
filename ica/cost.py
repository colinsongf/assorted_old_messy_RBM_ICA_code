#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

from numpy import *
from scipy.linalg import norm
from IPython import embed
from numdifftools import Derivative, Gradient



def negAvgTwoNorm_OLD(vv, XX, delta, kk):
    '''norm(vv) need not be 1.

    vv: filter column vector (1 dimensional)
    XX: data matrix, one example per column
    delta: how much to add to vk
    kk: scalar, which element to use to compute the derivative dC / dv_k

    Notes on 5/27/13'''

    assert len(vv.shape) == 1
    vvv = vv.copy()
    vvv[kk] += delta

    NN = XX.shape[1]
    ww = vvv / norm(vvv)
    act = dot(XX.T, ww)

    cost = -1.0 / NN * (act**2).sum()

    #dwk_dvk = 1.0 / norm(vvv) - vvv[kk] / norm(vvv)**3       # off by factor of norm(vvv)
    #dwk_dvk = 1.0 / norm(vvv)
    #dwk_dvk = vvv[kk] / norm(vvv)**3

    dwk_dvk = 1.0 / norm(vvv) - vvv[kk]**2 / norm(vvv)**3

    dc_dwk  = -2.0 / NN * (act * XX[kk,:]).sum()
    dc_dvk  = dc_dwk * dwk_dvk

    #embed()
    
    return cost, dc_dvk



def negAvgTwoNorm_vec(vv, XX):
    '''norm(vv) need not be 1.

    vv: filter column vector (1 dimensional)
    XX: data matrix, one example per column

    Notes on 5/27/13'''

    assert len(vv.shape) == 1

    NN = XX.shape[1]
    ww = vv / norm(vv)
    act = dot(XX.T, ww)

    cost = -1.0 / NN * (act**2).sum()

    grad_c_wrtw  = -2.0 / NN * dot(XX, act)

    # remove radial portion of gradient
    grad = grad_c_wrtw - ww * dot(grad_c_wrtw, ww)    # norm(ww) = 1
    
    # shink or grow by relative size of vv vs ww
    grad /= norm(vv)

    return cost, grad



def negAvgTwoNorm_mat(VV, XX):
    '''norm(vv) need not be 1.

    VV: filter matrix, one filter per column
    XX: data matrix, one example per column

    Notes on 5/27/13'''

    assert len(VV.shape) == 2
    assert len(XX.shape) == 2

    NN = XX.shape[1]
    colNormsVV = sqrt((VV**2).sum(0))
    WW = VV / colNormsVV
    act = dot(WW.T, XX)

    cost = -1.0 / NN * (act**2).sum()

    grad_c_wrtw  = -2.0 / NN * dot(XX, act.T)

    # remove radial portion of gradient
    grad = grad_c_wrtw - (WW * (WW * grad_c_wrtw).sum(0))         # norm(cols of WW) = 1
    
    # shink or grow by relative size of vv vs ww
    grad /= colNormsVV

    return cost, grad



def negAvgTwoNorm_matflat(VVflat, XX, VVshape):
    return negAvgTwoNorm_mat(reshape(VVflat, VVshape), XX)



def wwOfvv(vv):
    '''
    In: vector
    Out: vector
    '''
    ww = vv / norm(vv)
    return ww

    

def check_negAvgTwoNorm_vec():
    random.seed(0)
    dim = 100
    NN  = 1000
    XX = random.normal(0, 1, (dim,NN))
    vv = random.normal(0, 1, (dim,))

    kk = 0

    cost,anaGrad = negAvgTwoNorm_vec(vv, XX)
    gradFn = Gradient(lambda v: negAvgTwoNorm_vec(v, XX)[0])
    numGrad = gradFn(vv)

    if any(abs(anaGrad - numGrad) > 10 * gradFn.error_estimate) or gradFn.error_estimate.max() > 1e-10:
        print 'Possible failure!'

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad) / gradFn.error_estimate
    else:
        print 'Test passed! (max error: %s)' % abs(anaGrad - numGrad).max()



def check_negAvgTwoNorm_mat():
    random.seed(0)
    dim = 100
    NN  = 1000
    kk = 400
    XX = random.normal(0, 1, (dim,NN))
    VV = random.normal(0, 1, (dim,kk))

    kk = 0

    cost,anaGrad = negAvgTwoNorm_mat(VV, XX)

    #embed()
    
    gradFn = Gradient(lambda v: negAvgTwoNorm_matflat(v, XX, VV.shape)[0])
    numGrad = gradFn(VV.flatten())
    numGrad = reshape(numGrad, VV.shape)

    if any(abs(anaGrad - numGrad) > 10 * gradFn.error_estimate) or gradFn.error_estimate.max() > 1e-10:
        print 'Possible failure!'

        print 'cost:', cost
        print '\nanalytical grad:', anaGrad
        print '\nnumerical grad:', numGrad
        print '\nanalytical / numerical:', anaGrad / numGrad

        print 'largest error estimate:', gradFn.error_estimate.max()
        print 'errors / error_estimate:', abs(anaGrad - numGrad) / gradFn.error_estimate
    else:
        print 'Test passed! (max error: %s)' % abs(anaGrad - numGrad).max()




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
    check_negAvgTwoNorm_vec()
    #check_negAvgTwoNorm_mat()
    check_vs_vec_negAvgTwoNorm_mat()


if __name__ == '__main__':
    main()
