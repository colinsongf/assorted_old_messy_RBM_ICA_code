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



def negAvgTwoNorm_matflat(VVflat, XX, VVshape):
    return negAvgTwoNorm_mat(reshape(VVflat, VVshape), XX)



