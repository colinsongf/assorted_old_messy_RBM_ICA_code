#! /usr/bin/env python

from numpy import tanh



def sigmoid(xx):
    '''Compute the logistic/sigmoid in a numerically stable way (using tanh).'''
    #return 1. / (1 + exp(-xx))
    #print 'returning', .5 * (1 + tanh(xx / 2.))
    return .5 * (1 + tanh(xx / 2.))
