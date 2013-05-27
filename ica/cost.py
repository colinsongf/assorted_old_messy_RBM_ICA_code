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



def negAvgTwoNorm(vv, XX):
    '''norm(vv) need not be 1.

    vv: filter column vector (1 dimensional)
    XX: data matrix, one example per column

    Notes on 5/27/13'''

    assert len(vv.shape) == 1

    NN = XX.shape[1]
    ww = vv / norm(vv)
    act = dot(XX.T, ww)

    cost = -1.0 / NN * (act**2).sum()

    #dwk_dvk = 1.0 / norm(vvv) - vvv[kk] / norm(vvv)**3       # off by factor of norm(vvv)
    #dwk_dvk = 1.0 / norm(vvv)
    #dwk_dvk = vvv[kk] / norm(vvv)**3

    grad_c_wrtw  = -2.0 / NN * dot(XX, act)

    # remove radial portion of gradient
    grad = grad_c_wrtw - ww * dot(ww, grad_c_wrtw / norm(grad_c_wrtw))
    # shink or grow by relative size of vv vs ww
    grad /= norm(vv)

    return cost, grad



def wwOfvv(vv):
    '''
    In: vector
    Out: vector
    '''
    ww = vv / norm(vv)
    return ww

    

def main():
    random.seed(0)
    dim = 100
    NN  = 1000
    XX = random.normal(0, 1, (dim,NN))
    vv = random.normal(0, 1, (dim,))

    kk = 0

    cost,anaGrad = negAvgTwoNorm(vv, XX)
    print 'cost:', cost
    print 'analytical grad:', anaGrad

    gradFn = Gradient(lambda v: negAvgTwoNorm(v, XX)[0])
    numGrad = gradFn(vv)
    print 'numerical:', numGrad

    print 'anaDiv / numDiv =', anaDiv / numDiv
    print 'norm(vv)/2 =', norm(vv)/2
    

    gradfn = Gradient(lambda xx : negAvgTwoNorm(xx, XX))
    
    anaDiv = negAvgTwoNorm(vv, XX, 0, kk)[1]
    print 'returned:', negAvgTwoNorm(vv, XX, 0, kk)

    dfun = Derivative(lambda delta : negAvgTwoNorm(vv, XX, delta, kk)[0])
    numDiv = dfun(0)
    print 'numerical:', numDiv

    print 'anaDiv / numDiv =', anaDiv / numDiv
    print 'norm(vv)/2 =', norm(vv)/2
    
    #embed()


if __name__ == '__main__':
    main()
