#! /usr/bin/env ipythonpl

'''
Research code

Jason Yosinski
'''

import pdb
import os, sys, time
from numpy import *
from PIL import Image
#from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg') # plot with no display
from matplotlib import pyplot

from util.ResultsManager import resman, fmtSeconds
from util.plotting import tile_raster_images, pil_imagesc
from util.dataLoaders import loadFromPklGz, saveToFile
from util.math import sigmoid
from rbm.pca import PCA



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
    def __init__(self, imgShape, lambd = .005, nFeatures = 800, epsilon = 1e-5, float32 = False, saveDir = ''):
        self.lambd      = lambd
        self.nFeatures  = nFeatures
        self.epsilon    = epsilon
        self.saveDir    = saveDir
        self.imgShape   = imgShape
        self.imgIsColor = len(imgShape) > 2
        self.nInputDim  = prod(self.imgShape)
        self.float32    = float32
        self.costLog    = None

    def cost(self, WW, data, plotEvery = None):
        nInputDim = data.shape[0]
        nDatapoints = data.shape[1]
        if self.nInputDim != nInputDim:
            raise Exception('Expected shape %s = %d dimensional input, but got %d' % (repr(self.imgShape), self.nInputDim, nInputDim))

        # NOTE: Flattening and reshaping is in C order in numpy but Fortran order in Matlab. This should not matter.
        WW = WW.reshape(self.nFeatures, nInputDim)
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
        WGrad /= nDatapoints

        thislog = array([sparsityCost, reconstructionCost, cost])
        if isinstance(self.costLog, ndarray):
            self.costLog = vstack((self.costLog, thislog))
        else:
            self.costLog = thislog

        print 'f =', cost, '|grad| =', linalg.norm(grad)

        # Return cost and grad per data point
        return cost, grad


    def dataPrep(self, data, whiten, normData):
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

        if self.saveDir:
            pil_imagesc(cov(data),
                        saveto = os.path.join(self.saveDir, 'dataCov_0raw.png'))

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
            pil_imagesc(cov(data),
                        saveto = os.path.join(self.saveDir, 'dataCov_1prenorm.png'))
        if normData:
            # Project each patch to the unit ball
            patchNorms = sqrt(sum(data**2, 0) + (1e-8))
            data = data / patchNorms
        if self.saveDir:
            pil_imagesc(cov(data),
                        saveto = os.path.join(self.saveDir, 'dataCov_2postnorm.png'))

        return data


    def runOptimization(self, data, maxFun, plotEvery):
        # Initialize weights WW
        WW = random.randn(self.nFeatures, self.nInputDim)
        WW = (WW.T / sqrt(sum(WW ** 2, 1))).T
        WW = WW.flatten()

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
        #xopt, fval, info = fmin_l_bfgs_b(lambda WW : self.cost(WW, data),
        #                                 WW,
        #                                 fprime = None,   # function call returns value and gradient
        #                                 approx_grad = False,
        #                                 iprint = 1,
        #                                 factr = 1e3,
        #                                 maxfun = maxFun)
        results = minimize(lambda WW : self.cost(WW, data, plotEvery),
                                    WW,
                                    jac = True,    # const function retuns both value and gradient
                                    method = 'L-BFGS-B',
                                    options = {'maxiter': maxFun, 'disp': True})
        fval = results['fun']
        wallSeconds = time.time() - startWall
        print 'Optimization results:'
        for key in ['status', 'nfev', 'success', 'fun', 'message']:
            print '  %20s: %s' % (key, results[key])
        print '  %20s: %s' % ('fval', fval)
        print '  %20s: %s' % ('fval/example', fval/data.shape[1])
        print '  %20s: %s' % ('wall time', fmtSeconds(wallSeconds))
        print '  %20s: %s' % ('wall time/funcall', fmtSeconds(wallSeconds / results['nfev']))

        WW = results['x'].reshape(self.nFeatures, self.nInputDim)

        # Renormalize each patch of WW back to unit ball
        WW = (WW.T / sqrt(sum(WW**2, axis=1))).T

        return WW


    def plotCostLog(self):
        # plot sparsity/reconstruction costs over time
        costs = self.costLog
        self.costLog = None
        pyplot.plot(costs[:,0], 'b-', costs[:,1], 'r-')
        pyplot.hold(True)
        pyplot.plot(costs[:,2], '--', color = (.7,0,.7,1))
        pyplot.legend(('sparsity * %s' % repr(self.lambd), 'reconstruction', 'total'))
        pyplot.xlabel('iteration'); pyplot.ylabel('cost')
        if self.saveDir:
            pyplot.savefig(os.path.join(self.saveDir, 'cost.png'))
            pyplot.savefig(os.path.join(self.saveDir, 'cost.pdf'))


    def getXYNumTiles(self):
        tilesX = int(sqrt(self.nFeatures * 2./3))
        tilesY = self.nFeatures / tilesX
        return tilesX, tilesY

    
    def plotWW(self, WW, filePrefix = 'WW'):
        if self.saveDir:
            tilesX, tilesY = self.getXYNumTiles()
            image = Image.fromarray(tile_raster_images(
                X = WW,
                img_shape = self.imgShape, tile_shape = (tilesX,tilesY),
                tile_spacing=(1,1),
                scale_colors_together = True))
            image.save(os.path.join(self.saveDir, '%s.png' % filePrefix))
            if self.imgIsColor:
                image = Image.fromarray(tile_raster_images(
                    X = WW,
                    img_shape = self.imgShape, tile_shape = (tilesX,tilesY),
                    tile_spacing=(1,1),
                    scale_colors_together = False))
                image.save(os.path.join(self.saveDir, '%s_rescale_indiv.png' % filePrefix))


    def plotActivations(self, WW, data):
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


    def plotReconstructions(self, WW, data, number = 20):
        '''Plots reconstructions for some randomly chosen data points.'''

        if self.saveDir:
            hidden = dot(WW, data[:,:number])
            reconstruction = dot(WW.T, hidden)
            
            tilesX, tilesY = self.getXYNumTiles()
            for ii in xrange(number):
                hilightAmount = abs(hidden[:,ii])
                hilightAmount -= hilightAmount.min()
                hilightAmount /= hilightAmount.max() + 1e-6
                hilights = outer(hilightAmount, array([1,0,0]))
                tileImg = Image.fromarray(tile_raster_images(
                    X = WW,
                    img_shape = self.imgShape, tile_shape = (tilesX,tilesY),
                    tile_spacing=(2,2),
                    scale_colors_together = True,
                    hilights = hilights))
                rawReconErr = array([data[:,ii], reconstruction[:,ii], reconstruction[:,ii]-data[:,ii]])
                rawReconErrImg = Image.fromarray(tile_raster_images(
                    X = rawReconErr,
                    img_shape = self.imgShape, tile_shape = (rawReconErr.shape[0], 1),
                    tile_spacing=(1,1),
                    scale_colors_together = True))
                rescaleFactor = 2
                rawReconErrImg = rawReconErrImg.resize([x*rescaleFactor for x in rawReconErrImg.size])
                size = (tileImg.size[0] + rawReconErrImg.size[0] + rescaleFactor,
                        max(tileImg.size[1], rawReconErrImg.size[1]))
                wholeImage = Image.new('RGBA', size, (51, 51, 51))
                wholeImage.paste(tileImg, (0, 0))
                wholeImage.paste(rawReconErrImg, (tileImg.size[1] + rescaleFactor, 0))
                wholeImage.save(os.path.join(self.saveDir, '%s_%d.png' % ('testing_recon', ii)))


    def run(self, data, maxFun = 300, whiten = False, normData = True, plotEvery = None):
        '''data should be one data point per COLUMN! (different)'''

        data = self.dataPrep(data, whiten = whiten, normData = normData)

        WW = self.runOptimization(data, maxFun, plotEvery)

        if self.saveDir:
            saveToFile(os.path.join(self.saveDir, 'WW.pkl.gz'), WW)

        # Make and save some plots
        self.plotCostLog()
        self.plotWW(WW)
        self.plotActivations(WW, data)
        self.plotReconstructions(WW, data)



if __name__ == '__main__':
    resman.start('junk', diary = False)

    data = loadFromPklGz('../data/rica_hyv_patches_16.pkl.gz')
    #data = data[:,:5000]  #HACK
    
    random.seed(0)
    rica = RICA(imgShape = (16, 16),
                nFeatures = 400,
                lambd = .05,
                epsilon = 1e-5,
                float32 = False,
                saveDir = resman.rundir)
    rica.run(data, plotEvery = None, maxFun = 30)

    resman.stop()
