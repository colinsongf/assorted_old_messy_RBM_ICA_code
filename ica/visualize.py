#! /usr/bin/env python


'''
Research code

Jason Yosinski

Higher level utilities for visualizing learning progress. See also
lower level util.plotting.
'''

import os
import sys
import ipdb as pdb
from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm
from numpy import *

from util.plotting import tile_raster_images, pil_imagesc, scale_some_rows_to_unit_interval, scale_rows_together_to_unit_interval
from util.cache import cached



def plotImageData(data, imgShape, saveDir = None, prefix = 'imgdata', tileShape = (20,30)):
    isColor = (len(imgShape) > 2)
    if saveDir:
        image = Image.fromarray(tile_raster_images(
            X = data.T, img_shape = imgShape,
            tile_shape = tileShape, tile_spacing=(1,1),
            scale_rows_to_unit_interval = False))
        image.save(os.path.join(saveDir, '%s.png' % prefix))
        image = Image.fromarray(tile_raster_images(
            X = data.T, img_shape = imgShape,
            tile_shape = tileShape, tile_spacing=(1,1),
            scale_rows_to_unit_interval = True,
            scale_colors_together = True))
        image.save(os.path.join(saveDir, '%s_rescale.png' % prefix))
        if isColor:
            image = Image.fromarray(tile_raster_images(
                X = data.T, img_shape = imgShape,
                tile_shape = tileShape, tile_spacing=(1,1),
                scale_rows_to_unit_interval = True,
                scale_colors_together = False))
            image.save(os.path.join(saveDir, '%s_rescale_indiv.png' % prefix))



def plotCov(data, saveDir = None, prefix = 'imgdata'):
    if saveDir:
        cv = cached(cov, data)
        #cv = cov(data)
        pil_imagesc(cv, saveto = os.path.join(saveDir, '%s_cov.png' % prefix))



def getTileShape(number):
    tilesX = int(sqrt(number * 2./3))
    tilesY = number / tilesX
    return tilesX, tilesY



def plotImageRicaWW(WW, imgShape, saveDir, tileShape = None, prefix = 'WW'):
    imgIsColor = len(imgShape) > 2
    nOutputs, nInputs = WW.shape
    if tileShape is None: tileShape = getTileShape(nOutputs)

    if saveDir:
        image = Image.fromarray(tile_raster_images(
            X = WW,
            img_shape = imgShape, tile_shape = tileShape,
            tile_spacing=(1,1),
            scale_colors_together = True))
        image.save(os.path.join(saveDir, '%s.png' % prefix))
        if imgIsColor:
            image = Image.fromarray(tile_raster_images(
                X = WW,
                img_shape = imgShape, tile_shape = tileShape,
                tile_spacing=(1,1),
                scale_colors_together = False))
            image.save(os.path.join(saveDir, '%s_rescale_indiv.png' % prefix))



def plotRicaActivations(WW, data, saveDir = None, prefix = 'activations'):
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
            pyplot.savefig(os.path.join(saveDir, '%s_data_hist_%03d.png' % (prefix, ii)))
        for ii in range(10):
            pyplot.clf()
            pyplot.hist(hiddenActivationsRandom[:,ii])
            pyplot.savefig(os.path.join(saveDir, '%s_rand_hist_%03d.png' % (prefix, ii)))

    if saveDir:
        image = Image.fromarray((hiddenActivationsData.T + 1) * 128).convert('L')
        image.save(os.path.join(saveDir, '%s_data.png' % prefix))
        image = Image.fromarray((hiddenActivationsRandom.T + 1) * 128).convert('L')
        image.save(os.path.join(saveDir, '%s_random.png' % prefix))



def plotRicaReconstructions(rica, data, imgShape, saveDir = None, unwhitener = None, tileShape = None, number = 50, prefix = 'recon', onlyHilights = False, hilightCmap = None):
    '''Plots reconstructions for some randomly chosen data points.'''

    if saveDir:
        print 'Plotting %d recon plots...' % number,
        sys.stdout.flush()
        imgIsColor = len(imgShape) > 2
        nOutputs, nInputs = rica.WW.shape
        if tileShape is None: tileShape = getTileShape(nOutputs)
        tileRescaleFactor  = 2
        reconRescaleFactor = 3
        
        font = ImageFont.load_default()

        hidden = dot(rica.WW, data[:,:number])
        reconstruction = dot(rica.WW.T, hidden)

        if unwhitener:
            #pdb.set_trace() DEBUG?
            dataOrig = unwhitener(data[:,:number])
            reconstructionOrig = unwhitener(reconstruction[:,:number])
        for ii in xrange(number):
            # Hilighted tiled image
            hilightAmount = abs(hidden[:,ii])
            maxHilight = hilightAmount.max()
            #hilightAmount -= hilightAmount.min()   # Don't push to 0
            hilightAmount /= maxHilight + 1e-6

            if hilightCmap:
                cmap = cm.get_cmap(hilightCmap)
                hilights = cmap(hilightAmount)[:,:3]  # chop off alpha channel
            else:
                # default black -> red colormap
                hilights = outer(hilightAmount, array([1,0,0]))
            
            tileImg = Image.fromarray(tile_raster_images(
                X = rica.WW,
                img_shape = imgShape, tile_shape = tileShape,
                tile_spacing=(2,2),
                scale_colors_together = True,
                hilights = hilights,
                onlyHilights = onlyHilights))
            tileImg = tileImg.resize([x*tileRescaleFactor for x in tileImg.size])

            # Input / Reconstruction image
            if unwhitener:
                rawReconErr = array([dataOrig[:,ii], data[:,ii], reconstruction[:,ii], reconstructionOrig[:,ii],
                                     reconstruction[:,ii]-data[:,ii], reconstructionOrig[:,ii]-dataOrig[:,ii]])
                # Scale data-raw and recon-raw together between 0 and 1
                rawReconErr = scale_rows_together_to_unit_interval(rawReconErr, [0, 3], anchor0 = False)
                # Scale data-white and recon-white together, map 0 -> 50% gray
                rawReconErr = scale_rows_together_to_unit_interval(rawReconErr, [1, 2], anchor0 = True)
                # Scale diffs independently to [0,1]
                rawReconErr = scale_some_rows_to_unit_interval(rawReconErr, [4, 5])
            else:
                rawReconErr = array([data[:,ii], reconstruction[:,ii],
                                     reconstruction[:,ii]-data[:,ii]])
                # Scale data-raw and recon-raw together between 0 and 1
                rawReconErr = scale_rows_together_to_unit_interval(rawReconErr, [0, 1], anchor0 = False)
                # Scale diffs independently to [0,1]
                rawReconErr = scale_some_rows_to_unit_interval(rawReconErr, [2])
            rawReconErrImg = Image.fromarray(tile_raster_images(
                X = rawReconErr,
                img_shape = imgShape, tile_shape = (rawReconErr.shape[0], 1),
                tile_spacing=(1,1),
                scale_rows_to_unit_interval = False))
            rawReconErrImg = rawReconErrImg.resize([x*reconRescaleFactor for x in rawReconErrImg.size])

            # Add Red activation limit
            redString = '%g' % maxHilight
            fontSize = font.font.getsize(redString)
            size = (max(tileImg.size[0], fontSize[0]), tileImg.size[1] + fontSize[1])
            tempImage = Image.new('RGBA', size, (51, 51, 51))
            tempImage.paste(tileImg, (0, 0))
            draw = ImageDraw.Draw(tempImage)
            draw.text(((size[0]-fontSize[0])/2, size[1]-fontSize[1]), redString, font=font)
            tileImg = tempImage

            # Combined
            costEtc = rica.cost(rica.WW, data[:,ii:ii+1])
            costString = rica.getReconPlotString(costEtc)
            fontSize = font.font.getsize(costString)
            size = (max(tileImg.size[0] + rawReconErrImg.size[0] + reconRescaleFactor, fontSize[0]),
                    max(tileImg.size[1], rawReconErrImg.size[1]) + fontSize[1])
            wholeImage = Image.new('RGBA', size, (51, 51, 51))
            wholeImage.paste(tileImg, (0, 0))
            wholeImage.paste(rawReconErrImg, (tileImg.size[0] + reconRescaleFactor, 0))
            draw = ImageDraw.Draw(wholeImage)
            draw.text(((size[0]-fontSize[0])/2, size[1]-fontSize[1]), costString, font=font)
            wholeImage.save(os.path.join(saveDir, '%s_%04d.png' % (prefix, ii)))

        print 'done.'



def plotTopActivations(activations, data, imgShape, saveDir = None, nActivations = 50, nSamples = 20, prefix = 'topact'):
    '''Plots top and bottom few activations for the first number activations.'''

    if saveDir:
        sortIdx = argsort(activations, 1)

        plotData = zeros((prod(imgShape), nActivations*nSamples))

        for ii in range(nActivations):
            idx = sortIdx[ii,-1:-(nSamples+1):-1]
            plotData[:,(ii*nSamples):((ii+1)*nSamples)] = data[:,idx]

        image = Image.fromarray(tile_raster_images(
            X = plotData.T, img_shape = imgShape,
            tile_shape = (nActivations, nSamples), tile_spacing=(1,1),
            scale_rows_to_unit_interval = True))
        image.save(os.path.join(saveDir, '%s.png' % prefix))
