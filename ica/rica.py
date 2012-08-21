#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys
from numpy import *
from matplotlib import pyplot, mlab
from PIL import Image

from util.ResultsManager import resman
from util.plotting import tile_raster_images
from rbm.pca import PCA
from rbm.utils import load_mnist_data, saveToFile, looser

#from scipy.optimize import fmin_bfgs, minimize
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from util.dataLoaders import loadFromPklGz



def whiten1f(xx):
    pass



def l2RowScaled(xx):
    epsilon = 1e-5;
    return (xx.T / (sum(xx**2,1) + epsilon)).T



def l2RowScaledGrad(xx, yy, outerDeriv):
    epsilon = 1e-5;
    epsSumSq = sum(xx ** 2, 1) + epsilon
    l2Rows = sqrt(epsSumSq)

    grad = (outerDeriv.T / l2Rows).T - (yy.T * (sum(outerDeriv * xx, 1) / epsSumSq)).T
    return grad



class OneTwoFunction(object):
    '''Allows a function to be called that returns a tuple. '''

    def __init__(self, function):
        self.function = function
        self.nextCall = 1
        self.result = None

    def one(self, value):
        if self.nextCall != 1:
            raise Exception('Wrong call order; was expecting %d' % self.nextCall)
        self.lastValue = value
        self.result = self.function(value)
        self.nextCall = 2
        return self.result[0]

    def two(self, value):
        if self.nextCall != 2:
            raise Exception('Wrong call order; was expecting %d' % self.nextCall)
        if value != self.lastValue:
            raise Exception('Called two with a different value.')
        self.nextCall = 1
        return self.result[1]



class RICA(object):
    def __init__(self, imgDim, lambd = .005, nFeatures = 800, epsilon = 1e-5, saveDir = ''):
        self.lambd     = lambd
        self.nFeatures = nFeatures
        self.epsilon   = epsilon
        self.saveDir   = saveDir
        self.imgDim    = imgDim


    def cost(self, WW, data):
        nInputDim = data.shape[0]
        if self.imgDim ** 2 != nInputDim:
            raise Exception('Expected %d * %d = %d input, but got %d' % (self.imgDim, self.imgDim, self.imgDim**2, nInputDim))

        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nFeatures, nInputDim)

        WWold = WW
        WW = l2RowScaled(WW)

        # % Forward Prop
        # h = W*x;
        # r = W'*h;
        hidden = dot(WW, data)
        reconstruction = dot(WW.T, hidden)
        
        # % Sparsity Cost
        # K = sqrt(params.epsilon + h.^2);
        # sparsity_cost = params.lambda * sum(sum(K));
        # K = 1./K;
        KK = sqrt(self.epsilon + hidden ** 2)
        sparsityCost = self.lambd * sum(KK)

        # % Reconstruction Loss and Back Prop
        # diff = (r - x);
        # reconstruction_cost = 0.5 * sum(sum(diff.^2));
        # outderv = diff;
        reconDiff = reconstruction - data
        reconstructionCost = .5 * sum(reconDiff ** 2)
        outDeriv = reconDiff

        # % compute the cost comprised of: 1) sparsity and 2) reconstruction
        # cost = sparsity_cost + reconstruction_cost;
        cost = sparsityCost + reconstructionCost

        # % Backprop Output Layer
        # W2grad = outderv * h';
        W2Grad = dot(outDeriv, hidden.T)

        # % Backprop Hidden Layer
        # outderv = W * outderv;
        # outderv = outderv + params.lambda * (h .* K);
        outDeriv = dot(WW, outDeriv)
        outDeriv = outDeriv + self.lambd * (hidden * KK)

        # W1grad = outderv * x';
        # Wgrad = W1grad + W2grad';
        W1Grad = dot(outDeriv, data.T)
        WGrad = W1Grad + W2Grad.T

        # % unproject gradient for minFunc
        # grad = l2rowscaledg(Wold, W, Wgrad, 1);
        # grad = grad(:);
        grad = l2RowScaledGrad(WWold, WW, WGrad)
        grad = grad.flatten()

        return cost, grad


    def run(self, data, maxFun = 300):
        '''data should be one data point per COLUMN! (different)'''
        nInputDim = data.shape[0]

        # Initialize weights WW
        WW = random.randn(self.nFeatures, nInputDim)
        WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
        WW = WW.flatten()

        # Run optimization
        #xopt = fmin_bfgs(lambda WW : self.cost(WW, data), WW)
        #function = OneTwoFunction(lambda WW : self.cost(WW, data))

        #def costFn(xx):
        #    print 'costFn'
        #    return self.cost(xx, data)[0]
        #def gradFn(xx):
        #    print 'gradFn'
        #    return self.cost(xx, data)[1]
        
        #xopt = fmin_bfgs(costFn,
        #                 WW,
        #                 fprime = gradFn,
        #                 maxiter = 3)
        #xopt = minimize(lambda WW : self.cost(WW, data),
        #                WW,
        #                method = 'BFGS',
        #                jac = True,    # returned along with cost
        #                options = {'maxiter': 3, 'disp': True},
        #                )

        print 'Starting optimization, maximum function calls = ', maxFun
        xopt, fval, info = fmin_l_bfgs_b(lambda WW : self.cost(WW, data),
                                         WW,
                                         fprime = None,   # function call returns value and gradient
                                         approx_grad = False,
                                         iprint = 1,
                                         maxfun = maxFun)
        
        print 'OPT DONE'
        WW = xopt.reshape(self.nFeatures, nInputDim)

        image = Image.fromarray(tile_raster_images(
            X = WW,
            img_shape = (self.imgDim, self.imgDim), tile_shape = (24,33),
            tile_spacing=(1,1)))
        if self.saveDir:  image.save(os.path.join(self.saveDir, 'WW.png'))
        #image.show()



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')

    rica = RICA(imgDim = 16,
                saveDir = resman.rundir)
    rica.run(data, maxFun = 10)
    
    resman.stop()
