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
from util.misc import sigmoidAndDeriv01, sigmoidAndDeriv11



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



def unflatWrapper(xflat, func, xshape, *args):
    # This might not quite work? See Evernote [Tuesday, May 28, 2013,  1:42 pm]
    return func(reshape(xflat, xshape), *args)



def autoencoderRepresentation(W1, b1, XX):
    '''Paired with autoencoderCost.

    Returns the hidden layer representation of XX given weights (W1, b1). Uses sigmoid activation with range 0 to 1.'''

    # Forward prop
    a1 = XX
    
    z2 = (dot(W1, a1).T + b1).T
    a2 = sigmoid01(z2)

    return a2

def autoencoderCost(thetaFlat, XX, hiddenLayerSize, beta = 0, rho = .05):
    '''Cost for a single hidden layer autoencoder with sigmoid activation function. Uses sigmoid with range of 0 to 1.

    thetaFlat: (W1, b1, W2, b2).flatten()
    XX: one example per column'''

    dim, numExamples = XX.shape
    W1shape = (hiddenLayerSize, dim)
    b1shape = (hiddenLayerSize,)
    W2shape = (dim, hiddenLayerSize)
    b2shape = (dim,)

    begin = 0
    W1 = reshape(thetaFlat[begin:begin+prod(W1shape)], W1shape);    begin += prod(W1shape)
    b1 = reshape(thetaFlat[begin:begin+prod(b1shape)], b1shape);    begin += prod(b1shape)
    W2 = reshape(thetaFlat[begin:begin+prod(W2shape)], W2shape);    begin += prod(W2shape)
    b2 = reshape(thetaFlat[begin:begin+prod(b2shape)], b2shape);    begin += prod(W1shape)

    #seterr(all = 'raise')
    #pdb.set_trace()

    # Forward prop
    a1 = XX
    
    z2 = (dot(W1, a1).T + b1).T
    a2, dSig_dz2 = sigmoidAndDeriv01(z2)

    rho_hat = a2.mean(1)
    #print 'rho_hat', rho_hat
    klDiv = rho * (log(rho) - log(rho_hat)) + (1-rho) * log((1-rho) / (1-rho_hat))   # sparsity term
    #print 'klDiv', klDiv

    z3 = (dot(W2, a2).T + b2).T
    a3, dSig_dz3 = sigmoidAndDeriv01(z3)

    diffs = a3-XX

    cost = 1.0/numExamples * .5 * (diffs**2).sum() + beta * klDiv.sum()

    # Backward prop
    delta3 = diffs * dSig_dz3

    klDivDerivTerm = beta * (-rho/rho_hat + (1-rho)/(1-rho_hat))
    delta2 = dSig_dz2 * (dot(W2.T, delta3).T + klDivDerivTerm).T

    # Average gradients across examples
    W1grad = dot(delta2, a1.T)  / numExamples
    b1grad = delta2.sum(1)      / numExamples
    W2grad = dot(delta3, a2.T)  / numExamples
    b2grad = delta3.sum(1)      / numExamples
    
    grad = concatenate((W1grad.flatten(), b1grad.flatten(), W2grad.flatten(), b2grad.flatten()))

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



def check_autoencoderCost():
    ########### HERE
    random.seed(0)

    hiddenLayerSize = 2
    dim = 3
    numExamples = 4

    # random data
    XX = random.normal(.5, .25, (dim, numExamples))

    # random params
    W1 = random.normal(0, .1, (hiddenLayerSize, dim))
    b1 = random.normal(0, .1, (hiddenLayerSize,))
    W2 = random.normal(0, .1, (dim, hiddenLayerSize))
    b2 = random.normal(0, .1, (dim,))
    theta = concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))

    numericalCheckVectorGrad(autoencoderCost, theta, (XX, hiddenLayerSize, .1, .05))



def tests():
    # negAvgTwoNorm_*
    #check_negAvgTwoNorm_vec()
    #check_negAvgTwoNorm_mat()
    #check_negAvgTwoNorm_vecVmat()

    check_autoencoderCost()


if __name__ == '__main__':
    tests()

