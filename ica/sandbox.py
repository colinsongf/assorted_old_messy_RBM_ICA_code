#! /usr/bin/env ipythonpl

from matplotlib import *
from numpy import *
from scipy.linalg import norm
import Image
from IPython import embed
from scipy.optimize import minimize

from stackedLayers import *
from layers import *
from util.plotting import *



def nyu():
    dataLayer = NYU2_Labeled({'name': 'data',
                              'type': 'data',
                              'imageSize': (480,640),
                              'patchSize': (10,10),
                              'stride': (10,10),
                              'colorChannels': 1,
                              'depthChannels': 0})
    return dataLayer



def upson():
    dataLayer = UpsonData3({'name': 'data',
                            'type': 'data',
                            'imageSize': (240,320),
                            'patchSize': (10,10),
                            'stride': (10,10),
                            'colors': 1})
    return dataLayer



def excessAbs(filt, dat, negate = False):
    '''dat: examples are in columns'''
    nfilt = filt / norm(filt)   # don't give credit for making filt bigger or smaller norm
    dim,NN = dat.shape
    act = dot(dat.T, nfilt)
    actsq = act**2
    sig = sqrt(1.0/NN * actsq.sum())
    expectedAbs = sig * sqrt(2/pi)
    
    #actualAbs   = 1.0/NN * fabs(act).sum()
    actualAbs   = 1.0/NN * sqrt(actsq + 1e-6).sum()   # for 1e-6, same as above to within 1e-3
    
    excessAbs   = (actualAbs - expectedAbs) / expectedAbs
    #excessAbs   = (actualAbs - expectedAbs)
    #excessAbs   = actualAbs
    return excessAbs * (-1 if negate else 1)



def minafew(dat):
    dim,NN = dat.shape
    for ii in range(20):
        print '*' * 80
        print 'iter', ii
        filt = random.normal(0,1,(100,))
        filt /= norm(filt)
        results = minimize(excessAbs, filt, (dat,), method = 'L-BFGS-B', jac = False, options = {'maxiter': 200, 'disp': True})
        filtopt = results['x']
        filtopt /= norm(filtopt)
        imagesc(filt,    (10,10), peg0 = True)
        imagesc(filtopt, (10,10), peg0 = True)
