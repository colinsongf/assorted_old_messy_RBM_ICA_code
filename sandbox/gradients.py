#! /usr/bin/env python

from numpy import *
from numdifftools import Gradient



def fun(x):
    A = array([[1, 2], [3, 4]])
    b = array([3, 5])

    cost = .5 * dot(x.T, dot(A, x)) + dot(b, x)

    grad = .5 * dot(A + A.T, x) + b

    return cost, grad



def main():
    x0 = array([5, 7])
    print 'fun(x0)  =', fun(x0)
    dfun = Gradient(lambda x: fun(x)[0])
    print 'dfun(x0) =', dfun(x0)



if __name__ == '__main__':
    main()
