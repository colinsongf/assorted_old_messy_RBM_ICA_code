#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import ipdb as pdb
import os, sys, time
from numpy import *
#from PIL import Image, ImageFont, ImageDraw
#from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import minimize
from GitResultsManager import resman, fmtSeconds

import matplotlib
#matplotlib.use('Agg') # plot with no display
from matplotlib import pyplot

from util.plotting import tile_raster_images, pil_imagesc, scale_some_rows_to_unit_interval
from util.dataLoaders import loadFromPklGz, saveToFile
from util.math import sigmoid
from util.cache import cached



def l2RowScaled(xx, epsilon = 1e-5):
    return (xx.T / sqrt(sum(xx**2,1) + epsilon)).T



def l2RowScaledGrad(xx, yy, outerDeriv):
    # TODO: may still need to check this!
    epsilon = 1e-5;
    epsSumSq = sum(xx ** 2, 1) + epsilon
    l2Rows = sqrt(epsSumSq)

    grad = (outerDeriv.T / l2Rows).T - (yy.T * (sum(outerDeriv * xx, 1) / epsSumSq)).T
    return grad



class RICA(object):
    def __init__(self, nInputs, nOutputs = 400, lambd = .005, epsilon = 1e-5,
                 float32 = False, saveDir = '', initWW = True):
        self.nInputs    = nInputs
        self.nOutputs   = nOutputs
        self.lambd      = lambd
        self.epsilon    = epsilon
        self.saveDir    = saveDir
        #self.imgShape   = imgShape
        #self.imgIsColor = hasattr(imgShape, '__len__') and len(imgShape) > 2
        self.float32    = float32
        #self.doPlots    = doPlots
        self.costLog    = None
        #self.pca        = None    # Used for whitening / unwhitening data    #########

        self.WWshape    = (self.nOutputs, self.nInputs)
        self.WW         = None    # Not initialized yet
        if initWW:
            self.initWW()


    def initWW(self, seed = None):
        rng = random.RandomState(seed)      # if seed is None, this takes its seed from timer
        WW = rng.normal(0, 1, self.WWshape)
        self.WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
        #self.WW = WW.flatten()
        

    def costAndLog(self, WW, data, plotEvery = None):
        cost, sparsityCost, reconstructionCost, grad = self.cost(WW, data, plotEvery = plotEvery)

        thislog = array([sparsityCost, reconstructionCost, cost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        print 'f =', cost, '|grad| =', linalg.norm(grad)

        # Return cost and grad per data point
        return cost, grad


    def cost(self, WW, data, plotEvery = None):
        nInputs = data.shape[0]
        nDatapoints = data.shape[1]
        if self.nInputs != nInputs:
            raise Exception('Expected %d dimensional input, but got %d' % (self.nInputs, nInputs))

        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nOutputs, nInputs)
        WWold = WW
        WW = l2RowScaled(WW)

        numEvals = 0 if self.costLog is None else self.costLog.shape[0]
        if plotEvery and numEvals % plotEvery == 0:
            self.plotWW(WW, filePrefix = 'intermed_WW_%04d' % numEvals)

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
        KK = 1/KK

        # % Reconstruction Loss and Back Prop
        # diff = (r - x);
        # reconstruction_cost = 0.5 * sum(sum(diff.^2));
        # outderv = diff;
        reconDiff = reconstruction - data
        reconstructionCost = .5 * sum(reconDiff ** 2)
        outDeriv = reconDiff

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

        # % compute the cost comprised of: 1) sparsity and 2) reconstruction
        # cost = sparsity_cost + reconstruction_cost;
        #print '   sp', sparsityCost, 'rc', reconstructionCost
        sparsityCost /= nDatapoints
        reconstructionCost /= nDatapoints
        cost = sparsityCost + reconstructionCost
        #WGrad /= nDatapoints

        return cost, sparsityCost, reconstructionCost, grad


    def dataPrep(self, data, whiten, normData):
        raise Exception('Use other methods now')
        # deleted
        return data


    def runOptimization(self, data, maxFun, plotEvery):

        # Convert to float32 to be faster, if desired
        if self.float32:
            data = array(data, dtype='float32')
            WW = array(WW, dtype='float32')

        # HACK to make faster HACK
        #data = data[:,:8000]

        print 'RICA fitting with %d %d-dimensional data points' % (data.shape[1], data.shape[0])

        print 'Starting optimization, maximum function calls =', maxFun

        self.costLog = None
        startWall = time.time()
        #xopt, fval, info = fmin_l_bfgs_b(lambda WW : self.costAndLog(WW, data),
        #                                 WW,
        #                                 fprime = None,   # function call returns value and gradient
        #                                 approx_grad = False,
        #                                 iprint = 1,
        #                                 factr = 1e3,
        #                                 maxfun = maxFun)
        results = minimize(lambda ww : self.costAndLog(ww, data, plotEvery),
                           self.WW.flatten(),
                           jac = True,    # const function retuns both value and gradient
                           method = 'L-BFGS-B',
                           options = {'maxiter': maxFun, 'disp': True})
        #results = cached(minimize,
        #                 self.costAndLog,
        #                 WW,
        #                 (data, plotEvery),
        #                 jac = True,    # const function retuns both value and gradient
        #                 method = 'L-BFGS-B',
        #                 options = {'maxiter': maxFun, 'disp': True})
        
        fval = results['fun']
        wallSeconds = time.time() - startWall
        print 'Optimization results:'
        for key in ['status', 'nfev', 'success', 'fun', 'message']:
            print '  %20s: %s' % (key, results[key])
        print '  %20s: %s' % ('fval', fval)
        print '  %20s: %s' % ('fval/example', fval/data.shape[1])
        print '  %20s: %s' % ('wall time', fmtSeconds(wallSeconds))
        print '  %20s: %s' % ('wall time/funcall', fmtSeconds(wallSeconds / results['nfev']))

        WW = results['x'].reshape(self.nOutputs, self.nInputs)

        # Renormalize each patch of WW back to unit ball
        WW = (WW.T / sqrt(sum(WW**2, axis=1))).T

        return WW


    def plotCostLog(self):
        # plot sparsity/reconstruction costs over time
        costs = self.costLog
        #self.costLog = None    # disabled this reset.
        pyplot.figure()
        pyplot.plot(costs[:,0], 'b-', costs[:,1], 'r-')
        pyplot.hold(True)
        pyplot.plot(costs[:,2], '--', color = (.7,0,.7,1))
        pyplot.legend(('sparsity * %s' % repr(self.lambd), 'reconstruction', 'total'))
        pyplot.xlabel('iteration'); pyplot.ylabel('cost')
        if self.saveDir:
            pyplot.savefig(os.path.join(self.saveDir, 'cost.png'))
            pyplot.savefig(os.path.join(self.saveDir, 'cost.pdf'))
        pyplot.close()


    def getReconPlotString(self, costEtc):
        totalCost, sparsityCost, reconstructionCost, grad = costEtc
        return 'R: %g S*%g: %g T %g' % (reconstructionCost, self.lambd, sparsityCost, totalCost)


    def learn(self, data, maxFun = 300, whiten = False, normData = True, plotEvery = None):
        '''data should be one data point per COLUMN! (different)'''

        if self.WW is None:
            raise Exception('Initialize WW first!')
        if data.shape[0] != self.nInputs:
            raise Exception('Expected %d dimensional input, but got %d' % (self.nInputs, data.shape[0]))

        self.WW = self.runOptimization(data, maxFun, plotEvery)

        #if self.saveDir:
        #    saveToFile(os.path.join(self.saveDir, 'WW+pca.pkl.gz'), (WW, self.pca))

        # Make and save some plots
        self.plotCostLog()
        #if self.doPlots:
        #    self.plotWW(WW)
        #self.plotActivations(WW, data)
        #if self.doPlots:
        #    self.plotReconstructions(WW, data)



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    #data = data[:,:5000]  #HACK
    
    random.seed(0)
    rica = RICA(imgShape = (16, 16),
                nOutputs = 50,
                lambd = .05,
                epsilon = 1e-5,
                float32 = False,
                saveDir = resman.rundir)
    rica.learn(data, plotEvery = None, maxFun = 5, whiten = True)

    resman.stop()
