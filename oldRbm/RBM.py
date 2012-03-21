#! /usr/bin/env ipython -pylab

'''
Research code

Jason Yosinski
'''

from numpy import *
import pylab as p
from time import sleep
import copy
#from scipy import stats, mgrid, c_, reshape, random, rot90
import pdb

#import psyco
#psyco.full()



def logistic(xx):
    '''Compute the logistic in a numerically stable way (using tanh).'''
    #return 1. / (1 + exp(-xx))
    #print 'returning', .5 * (1 + tanh(xx / 2.))
    return .5 * (1 + tanh(xx / 2.))


class RBM(object):
    '''
    Implements a Restricted Boltzmann Machine
    '''

    def __init__(self, sizeV, sizeH):
        '''Construct an RBM'''

        self._sizeV = sizeV
        self._sizeH = sizeH

        self.v = (random.randint(0, 2, (self._sizeV, 1)) > .5) + 0
        self.h = (random.randint(0, 2, (self._sizeH, 1)) > .5) + 0

        self._W = random.normal(0, 1, (self._sizeH + 1, self._sizeV + 1))
        # Important: set 0,0 term to 0 (bias-bias term should not influence network)
        self._W[0,0] = 0

        self._reconErrorNorms = array([])


    def getV(self, withBias = False):
        '''Visible nodes'''
        if withBias:
            return self._v[0:]
        else:
            return self._v[1:]

    def setV(self, val):
        assert (len(val) == self._sizeV)
        self._v = array([hstack((array([1]), val.squeeze()))]).T

    v = property(getV, setV)


    def getH(self, withBias = False):
        '''Hidden nodes'''
        if withBias:
            return self._h[0:]
        else:
            return self._h[1:]

    def setH(self, val):
        assert (len(val) == self._sizeH)
        self._h = array([hstack((array([1]), val.squeeze()))]).T

    h = property(getH, setH)


    def getW(self):
        '''Combined weight and bias matrix'''
        return self._W

    def setW(self, val):
        assert (val.shape[0] == self._sizeH + 1)
        assert (val.shape[1] == self._sizeV + 1)
        self._W = val
        self._W[0,0] = 0

    W = property(getW, setW)


    def getReconErrorNorms(self):
        '''Vector of reconstruction error norms for each training vector seen so far.'''
        return self._reconErrorNorms

    reconErrorNorms = property(getReconErrorNorms)


    def energy(self):
        '''Compute energy of the network given the visible states,
        hidden states, and current extended weight matrix.

        #vv and hh must be column vectors.

        #To account for biases, vv[0,0] and hh[0,0] must be 1, and WW[0,0]
        #must be 0.'''

        assert (self._v[0,0] == 1)
        assert (self._h[0,0] == 1)
        assert (self._W[0,0] == 0)

        return -dot(dot(self._W, self._v).T, self._h)


    def v2h(self):
        '''Do a visible to hidden step.'''
        self._h = dot(self._W, self._v)
        self._h = (logistic(self._h) > random.uniform(0, 1, self._h.shape)) + 0
        self._h[0,0] = 1    # Bias term


    def h2v(self, activation = 'logisticBinary', param = 1, returnNoisefree = False):
        '''Do a hidden to visible step.'''
        ret = None
        self._v = dot(self._W.T, self._h)
        if activation == 'logisticBinary':
            self._v = (logistic(self._v) > random.uniform(0, 1, self._v.shape)) + 0
        elif activation == 'gaussianReal':
            if returnNoisefree:
                ret = copy.copy(self._v)
            self._v += param * random.normal(0, 1, self._v.shape)
        else:
            self._v[0,0] = 1    # Bias term
            raise Exception('Unknown activation: %s' % activation)
        self._v[0,0] = 1    # Bias term

        return ret


    def learn1(self, vv, epsilon = .1, activationH2V = 'logisticBinary', param = 1):
        '''Shift weights to better reconstruct the given visible vector.'''

        self.v = vv
        self.v2h()
        vihjData  = dot(self._h, self._v.T)
        if activationH2V == 'gaussianReal':
            noisefreeV = self.h2v(activation = activationH2V,
                                  param      = param,
                                  returnNoisefree = True)
            self._reconErrorNorms = hstack((self._reconErrorNorms,
                                            linalg.norm(noisefreeV[1:].squeeze() - vv)))
        else:
            self.h2v(activation = activationH2V,
                     param      = param)
            self._reconErrorNorms = hstack((self._reconErrorNorms,
                                            linalg.norm(self._v[1:].squeeze() - vv)))
        self.v2h()
        vihjRecon = dot(self._h, self._v.T)
        self._W += epsilon * (vihjData - vihjRecon)

        #print 'self._W[0,0] was', self._W[0,0]
        self._W[0,0] = 0
        

    def reconErrorVec(self, vv, activationH2V = 'logisticBinary'):
        '''Performs a V2H step, an H2V step, and then reports the
        reconstruction error as a vector of differences.'''

        self.v = vv
        self.v2h()
        self.h2v(activation = activationH2V)
        return self._v[1:].squeeze() - vv


    def reconErrorNorm(self, vv, activationH2V = 'logisticBinary'):
        '''2 Norm of the reconErrorVec for vector vv.'''

        return linalg.norm(self.reconErrorVec(vv, activationH2V))


    def plot(self, nSubplots = 3, skipH = False, skipW = False, skipV = False):
        '''Plot the hidden layer, weight layer, and visible layers in
        the current figure.'''
        lclr = [1,.47,0]

        curSubplot = 1
        if not skipH:
            ax = p.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            p.imshow(self._h.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeH < 25:
                p.xticks(arange(self._h.shape[0])-.5)
                p.yticks(arange(self._h.shape[1])-.5)
            else:
                p.xticks([])
                p.yticks([])
            p.axvline(.5, color=lclr, linewidth=2)

        if not skipW:
            ax = p.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            p.imshow(self._W, cmap='gray', interpolation='nearest', vmin=-2, vmax=2)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeH < 25 and self._sizeV < 25:
                p.xticks(arange(self._W.shape[1])-.5)
                p.yticks(arange(self._W.shape[0])-.5)
            else:
                p.xticks([])
                p.yticks([])
            p.axvline(.5, color=lclr, linewidth=2)
            p.axhline(.5, color=lclr, linewidth=2)

        if not skipV:
            ax = p.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            p.imshow(self._v.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeV < 25:
                p.xticks(arange(self._v.shape[0])-.5)
                p.yticks(arange(self._v.shape[1])-.5)
            else:
                p.xticks([])
                p.yticks([])
            p.axvline(.5, color=lclr, linewidth=2)



def main():
    Nv = 20
    Nh = 20
    
    rbm = RBM(Nv, Nh)

    #p.figure(1)

    energies = array([[]])
    for ii in range(100):
        ee = rbm.energy()
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))

        if mod(ii, 5) == 0:
            p.clf()
            rbm.plot(nSubplots = 4)
            ax = p.subplot(4,1,4)
            p.plot(energies[0])
            p.show()
            sleep(.1)

        rbm.v2h()
        print '  v->h step'

        ee = rbm.energy()
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))
        
        if mod(ii, 5) == 999:
            p.clf()
            rbm.plot(nSubplots = 4)
            ax = p.subplot(4,1,4)
            p.plot(energies[0])
            p.show()
            sleep(.1)

        rbm.h2v()
        print '  h->v step'



if __name__ == '__main__':
    main()
