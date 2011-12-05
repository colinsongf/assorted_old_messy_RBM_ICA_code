#! /usr/bin/env python

from numpy import *
from copy import copy
import pylab as pl
from time import sleep



def uniq(seq):
    '''Returns a list with duplicates removed. Order is preserved.
    From: http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order'''
    
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]



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



def hypercubeSucessor(node):
    '''Helper function: returns all nodes that have one more one than the current node'''
    ret = []
    for ii in range(len(node)):
        if node[ii] == 0:
            ret.append(tuple(node[:ii] + (1,) + node[ii+1:]))
    return ret



def hypercubeCornersEdges(dim = 3):
    '''Returns a list of corners and edges for the unit hypercube of the given dimension'''
    if dim > 15:
        raise Exception('dimension is probably too high')
    rDimRange = range(dim)
    rDimRange.reverse()
    nCorners = 2**dim
    nEdges   = nCorners * dim / 2

    # Corners
    corners = zeros((nCorners, dim))
    for ii in range(nCorners):
        # set to bits of iith binary number
        corners[ii,:] = [(ii & (1 << jj)) > 0 for jj in rDimRange]

    # Edges
    nodes = [tuple(0 for ii in range(dim))]
    seen = set(nodes)
    ii = 0
    edge0 = zeros((nEdges, dim))
    edge1 = zeros((nEdges, dim))
    while len(nodes) > 0:
        parent = nodes.pop(0)
        children = hypercubeSucessor(parent)
        nodes.extend(children)
        nodes = uniq(nodes)
        for child in children:
            edge0[ii,:] = parent
            edge1[ii,:] = child
            ii += 1
    #print 'ii =', ii, 'nodes is', nodes
    if ii != nEdges:
        raise Exception('Logic error.')

    return corners, (edge0, edge1)



def main():
    N = 5
    mat = random.random((N, N))
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

    corners, (edges0, edges1) = hypercubeCornersEdges(15)
    print corners, 'corners'
    print edges0, 'edges0'
    print edges1, 'edges1'



if __name__ == '__main__':
    main()
