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

from gradCheck import numericalCheckVectorGrad, numericalCheckMatrixGrad, checkVectorMatrixGradientsEqual



############################
#
#  Cost functions
#
############################



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



############################
#
#  Gradient checks
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

def check_negAvgTwoNorm_vecVmat():
    random.seed(0)
    dim = 100
    NN  = 1000
    kk = 400
    XX = random.normal(0, 1, (dim,NN))
    VV = random.normal(0, 1, (dim,kk))

    checkVectorMatrixGradientsEqual(negAvgTwoNorm_vec, negAvgTwoNorm_mat, VV, (XX,))



def tests():
    # negAvgTwoNorm_*
    check_negAvgTwoNorm_vec()
    check_negAvgTwoNorm_mat()
    check_negAvgTwoNorm_vecVmat()



if __name__ == '__main__':
    tests()

