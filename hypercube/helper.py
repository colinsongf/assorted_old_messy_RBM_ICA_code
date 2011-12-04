#! /usr/bin/env python

from numpy import *
import pylab as pl
from time import sleep
import pdb



def gramSchmidt(mat, det = False, inplace = True):
    '''Takes a matrix and returns an orthonormal version of it'''
    if mat.shape[0] != mat.shape[1]:
        raise Exception('gramSchmidt only works for square matrices.')
    if linalg.det(mat) == 0:
        raise Exception('gramSchmidt only works for full rank matrices.')
    if inplace:
        ret = mat
    else:
        ret = copy(mat)

    for ii in range(ret.shape[0]):
        # for each row
        # Orthogonalize w.r.t previous rows
        for prevII in range(ii):
            # numerically stable version
            ret[ii,:] -= dot(ret[ii,:], ret[prevII,:]) * ret[prevII,:]
        # Normalize this row
        ret[ii,:] /= linalg.norm(ret[ii,:])

    return ret



def main():
    N = 5
    mat = random.rand(N, N)
    print 'matrix M is:'
    print mat
    gs = gramSchmidt(mat)
    print 'matrix G after gramSchmidt is:'
    print gs
    print
    print '                  det(G):', linalg.det(gs), '\tshould be +/-1'
    print '        det(dot(G, G.T)):', linalg.det(dot(gs, gs.T)), '\tshould be 1'
    print '  det(dot(G, G.T) - eye):', linalg.det(dot(gs, gs.T) - eye(N)), '\tshould be ~0'
    #print dot(gs, gs.T) - eye(N)


if __name__ == '__main__':
    main()
