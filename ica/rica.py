#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys, time
from numpy import *
from PIL import Image
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

import matplotlib
matplotlib.use('Agg') # plot with no display
from matplotlib import pyplot

from util.ResultsManager import resman, fmtSeconds
from util.plotting import tile_raster_images
from util.dataLoaders import loadFromPklGz, saveToFile
from util.math import sigmoid
from rbm.pca import PCA
from rbm.utils import imagesc



def whiten1f(xx):
    pass



def l2RowScaled(xx):
    epsilon = 1e-5;
    return (xx.T / sqrt(sum(xx**2,1) + epsilon)).T



def l2RowScaledGrad(xx, yy, outerDeriv):
    # TODO: may still need to check this!
    epsilon = 1e-5;
    epsSumSq = sum(xx ** 2, 1) + epsilon
    l2Rows = sqrt(epsSumSq)

    grad = (outerDeriv.T / l2Rows).T - (yy.T * (sum(outerDeriv * xx, 1) / epsSumSq)).T
    return grad



class RICA(object):
    def __init__(self, imgShape, lambd = .005, nFeatures = 800, epsilon = 1e-5, saveDir = ''):
        self.lambd      = lambd
        self.nFeatures  = nFeatures
        self.epsilon    = epsilon
        self.saveDir    = saveDir
        self.imgShape   = imgShape
        self.imgIsColor = len(imgShape) > 2
        self.nInputDim  = prod(self.imgShape)

    def cost(self, WW, data):
        nInputDim = data.shape[0]
        if self.nInputDim != nInputDim:
            raise Exception('Expected shape %s = %d dimensional input, but got %d' % (repr(self.imgShape), self.nInputDim, nInputDim))

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
        KK = 1/KK

        # % Reconstruction Loss and Back Prop
        # diff = (r - x);
        # reconstruction_cost = 0.5 * sum(sum(diff.^2));
        # outderv = diff;
        reconDiff = reconstruction - data
        reconstructionCost = .5 * sum(reconDiff ** 2)
        outDeriv = reconDiff

        # % compute the cost comprised of: 1) sparsity and 2) reconstruction
        # cost = sparsity_cost + reconstruction_cost;
        #print '   sp', sparsityCost, 'rc', reconstructionCost
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

        print 'f =', cost, '|grad| =', linalg.norm(grad)

        return cost, grad


    def run(self, data, maxFun = 300, whiten = False, normData = True):
        '''data should be one data point per COLUMN! (different)'''
        nInputDim = data.shape[0]

        if self.saveDir:
            image = Image.fromarray(tile_raster_images(
                X = data.T, img_shape = self.imgShape,
                tile_shape = (20, 30), tile_spacing=(1,1),
                scale_rows_to_unit_interval = False))
            image.save(os.path.join(self.saveDir, 'data_raw.png'))
            image = Image.fromarray(tile_raster_images(
                X = data.T, img_shape = self.imgShape,
                tile_shape = (20, 30), tile_spacing=(1,1),
                scale_rows_to_unit_interval = True,
                scale_colors_together = True))
            image.save(os.path.join(self.saveDir, 'data_raw_rescale.png'))
            if self.imgIsColor:
                image = Image.fromarray(tile_raster_images(
                    X = data.T, img_shape = self.imgShape,
                    tile_shape = (20, 30), tile_spacing=(1,1),
                    scale_rows_to_unit_interval = True,
                    scale_colors_together = False))
                image.save(os.path.join(self.saveDir, 'data_raw_rescale_indiv.png'))

        if whiten:
            pca = PCA(data.T)
            dataWhite = pca.toZca(data.T, epsilon = 1e-6).T

            if self.saveDir:
                pyplot.semilogy(pca.fracVar, 'o-')
                pyplot.title('Fractional variance in each dimension')
                pyplot.savefig(os.path.join(self.saveDir, 'fracVar.png'))
                pyplot.savefig(os.path.join(self.saveDir, 'fracVar.pdf'))
                pyplot.close()

            data = dataWhite

            if self.saveDir:
                image = Image.fromarray(tile_raster_images(
                    X = data.T, img_shape = self.imgShape,
                    tile_shape = (20, 30), tile_spacing=(1,1),
                    scale_rows_to_unit_interval = True,
                    scale_colors_together = True))
                image.save(os.path.join(self.saveDir, 'data_white_rescale.png'))
                if self.imgIsColor:
                    image = Image.fromarray(tile_raster_images(
                        X = data.T, img_shape = self.imgShape,
                        tile_shape = (20, 30), tile_spacing=(1,1),
                        scale_rows_to_unit_interval = True,
                        scale_colors_together = False))
                    image.save(os.path.join(self.saveDir, 'data_white_rescale_indiv.png'))

        if self.saveDir:
            imagesc(cov(data))
            pyplot.savefig(os.path.join(self.saveDir, 'dataCov_prenorm.png'))
        if normData:
            # Project each patch to the unit ball
            patchNorms = sqrt(sum(data**2, 0) + (1e-8))
            data = data / patchNorms
        if self.saveDir:
            imagesc(cov(data))
            pyplot.savefig(os.path.join(self.saveDir, 'dataCov_postnorm.png'))

        # Initialize weights WW
        WW = random.randn(self.nFeatures, nInputDim)
        WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
        #loadedWW = loadtxt('../octave-randTheta')
        #WW = loadedWW
        WW = WW.flatten()

        # HACK to make faster HACK
        #data = data[:,:8000]

        print 'RICA fitting with %d %d-dimensional data points' % (data.shape[1], data.shape[0])

        print 'Starting optimization, maximum function calls =', maxFun

        startWall = time.time()
        xopt, fval, info = fmin_l_bfgs_b(lambda WW : self.cost(WW, data),
                                         WW,
                                         fprime = None,   # function call returns value and gradient
                                         approx_grad = False,
                                         iprint = 1,
                                         factr = 1e3,
                                         maxfun = maxFun)
        wallSeconds = time.time() - startWall
        print 'Optimization results:'
        for key,val in info.iteritems():
            print '  %20s: %s' % (key, repr(val))
        print '  %20s: %s' % ('fval', fval)
        print '  %20s: %s' % ('fval/example', fval/data.shape[1])
        print '  %20s: %s' % ('wall time', fmtSeconds(wallSeconds))
        print '  %20s: %s' % ('wall time/funcall', fmtSeconds(wallSeconds / info['funcalls']))
        
        WW = xopt.reshape(self.nFeatures, nInputDim)
        if self.saveDir:  saveToFile(os.path.join(self.saveDir, 'WW.pkl.gz'), WW)

        if self.saveDir:
            tilesX = int(sqrt(self.nFeatures * 2./3))
            tilesY = self.nFeatures / tilesX
            image = Image.fromarray(tile_raster_images(
                X = WW,
                img_shape = self.imgShape, tile_shape = (tilesX,tilesY),
                tile_spacing=(1,1),
                scale_colors_together = True))
            image.save(os.path.join(self.saveDir, 'WW.png'))
            image = Image.fromarray(tile_raster_images(
                X = WW,
                img_shape = self.imgShape, tile_shape = (tilesX,tilesY),
                tile_spacing=(1,1),
                scale_colors_together = False))
            image.save(os.path.join(self.saveDir, 'WW_rescale_indiv.png'))

        # Activation histograms
        hiddenActivationsData = dot(WW, data[:,:200])
        randomData = random.randn(data.shape[0], 200)
        randNorms = sqrt(sum(randomData**2, 0) + (1e-8))
        randomData /= randNorms
        hiddenActivationsRandom = dot(WW, randomData)

        enableIndividualHistograms = False
        if enableIndividualHistograms:
            for ii in range(10):
                pyplot.clf()
                pyplot.hist(hiddenActivationsData[:,ii])
                pyplot.savefig(os.path.join(self.saveDir, 'hidden_act_data_hist_%03d.png' % ii))
            for ii in range(10):
                pyplot.clf()
                pyplot.hist(hiddenActivationsRandom[:,ii])
                pyplot.savefig(os.path.join(self.saveDir, 'hidden_act_rand_hist_%03d.png' % ii))

        if self.saveDir:
            image = Image.fromarray((hiddenActivationsData.T + 1) * 128).convert('L')
            image.save(os.path.join(self.saveDir, 'hidden_act_data.png'))
            image = Image.fromarray((hiddenActivationsRandom.T + 1) * 128).convert('L')
            image.save(os.path.join(self.saveDir, 'hidden_act_random.png'))



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    
    random.seed(0)
    rica = RICA(imgShape = (16, 16),
                nFeatures = 400,
                lambd = .05,
                epsilon = 1e-5,
                saveDir = resman.rundir)
    rica.run(data, maxFun = 300)

    resman.stop()
