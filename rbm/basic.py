#! /usr/bin/env ipython -pylab

'''
Test code
'''

from numpy import *
#from numpy.random import *
import pylab as p
from time import sleep
#from scipy import stats, mgrid, c_, reshape, random, rot90
import pdb

#import psyco
#psyco.full()



def logistic(xx):
    return 1. / (1 + exp(-xx))



def calcEnergy(vv, hh, WW):
    '''Compute energy of the network given the visible states vv,
    hidden states hh, and extended weight matrix WW.

    vv and hh must be column vectors.

    To account for biases, vv[0,0] and hh[0,0] must be 1, and WW[0,0]
    must be 0.'''

    assert (vv[0,0] == 1)
    assert (hh[0,0] == 1)
    assert (WW[0,0] == 0)

    return -dot(dot(WW, vv).T, hh)



def main():
    Nv = 30
    Nh = 30
    
    # randomly initialize layers
    layer0 = random.randint(0, 2, (Nv+1,1))
    layer0 = (layer0 > .5) + 0
    layer0[0,0] = 1    # Bias term

    W01    = random.normal(0, .8, (Nh+1, Nv+1))
    W01[0,0] = 0    # Important: set 0,0 term to 0 (bias-biast term should not influence network)

    layer1 = random.randint(0, 1, (Nh+1,1))
    layer1 = (layer1 > .5) + 0
    layer1[0,0] = 1


    p.figure(1)

    energies = array([[]])
    for ii in range(100):
        ee = calcEnergy(layer0, layer1, W01)
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))

        if mod(ii, 5) == -1:
            p.clf()
            plotLayers(layer0, W01, layer1, energies)
            p.show()
            sleep(.1)
        
        layer1 = dot(W01, layer0)
        layer1 = (logistic(layer1) > random.uniform(0, 1, layer1.shape)) + 0
        layer1[0,0] = 1    # Bias term
        print '  v->h step'


        ee = calcEnergy(layer0, layer1, W01)
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))

        if mod(ii, 5) == 0:
            p.clf()
            plotLayers(layer0, W01, layer1, energies)
            p.show()
            sleep(.1)
        
        layer0 = dot(W01.T, layer1)
        layer0 = (logistic(layer0) > random.uniform(0, 1, layer0.shape)) + 0
        layer0[0,0] = 1    # Bias term
        print '  h->v step'

        
        #print repr(layer1.T)
        #print repr(layer0.T)

    




def plotLayers(vv, WW, hh, energies):
    Nv = vv.shape[0]-1
    Nh = hh.shape[0]-1
    lclr = [1,.47,0]
    
    ax = p.subplot(4,1,1)
    p.imshow(hh.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if Nh < 25:
        p.xticks(arange(hh.shape[0])-.5)
        p.yticks(arange(hh.shape[1])-.5)
    else:
        p.xticks([])
        p.yticks([])
    p.axvline(.5, color=lclr, linewidth=2)
    
    ax = p.subplot(4,1,2)
    p.imshow(WW, cmap='gray', interpolation='nearest', vmin=-2, vmax=2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if Nh < 25 and Nv < 25:
        p.xticks(arange(WW.shape[1])-.5)
        p.yticks(arange(WW.shape[0])-.5)
    else:
        p.xticks([])
        p.yticks([])
    p.axvline(.5, color=lclr, linewidth=2)
    p.axhline(.5, color=lclr, linewidth=2)

    ax = p.subplot(4,1,3)
    p.imshow(vv.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if Nv < 25:
        p.xticks(arange(vv.shape[0])-.5)
        p.yticks(arange(vv.shape[1])-.5)
    else:
        p.xticks([])
        p.yticks([])
    p.axvline(.5, color=lclr, linewidth=2)

    ax = p.subplot(4,1,4)
    p.plot(energies[0])



if __name__ == "__main__":
    main()
