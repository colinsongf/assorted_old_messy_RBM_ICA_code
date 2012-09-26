#! /usr/bin/env python

'''
[JBY] Utilities for learning.
Some stuff copied from https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/utils.py
'''

''' This file contains different utility functions that are not connected 
in anyway to the networks presented in the tutorials, but rather help in 
processing the outputs into a more understandable way. 

For example ``tile_raster_images`` helps in generating a easy to grasp 
image from a set of samples or weights.
'''

import pdb
import numpy
import Image

def scale_to_unit_interval(ndar,eps=1e-8):
    ''' Scales all values in the ndarray ndar to be between 0 and 1 '''
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def scale_all_rows_to_unit_interval(ndar,eps=1e-8):
    ''' Scales each row in the 2D array ndar to be between 0 and 1 '''
    assert(len(ndar.shape) == 2)
    ndar = ndar.copy()
    ndar = (ndar.T - ndar.min(axis=1)).T
    ndar = (ndar.T / (ndar.max(axis=1)+eps)).T
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing = (0,0), 
                       scale_rows_to_unit_interval = True, scale_colors_together = False,
                       output_pixel_vals = True):
    '''
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    '''

    assert(len(img_shape) == 2 or (len(img_shape) == 3 and img_shape[2] == 3)) # grayscale or RGB color
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    isColor = len(img_shape) == 3

    #pdb.set_trace()

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isColor:
        # massage to expected tuple form
        #assert(X.shape[1] % 3 == 0)
        #nPerChannel = X.shape[1] / 3

        if output_pixel_vals:
            dt = 'uint8'
        else:
            dt = X.dtype
        if str(dt) not in ('uint8', 'float32'):
            raise Exception('color only worsk for uint8 or float32 dtype, not %s' % dt)

        if scale_rows_to_unit_interval and scale_colors_together:
            X = scale_all_rows_to_unit_interval(X)

        X = X.reshape(X.shape[0], img_shape[0], img_shape[1], img_shape[2])
        X = (X[:,:,:,0].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             X[:,:,:,1].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             X[:,:,:,2].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             None, #numpy.ones(X[:,0:nPerChannel].shape, dtype=dt), # hardcode complete opacity
             )


    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            #out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            out_array = numpy.ones((out_shape[0], out_shape[1], 4), dtype='uint8') * 51
        else:
            #out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)
            out_array = numpy.ones((out_shape[0], out_shape[1], 4), dtype=X.dtype) * .2

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [51,51,51,255]
        else:
            channel_defaults = [.2,.2,.2,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype (unless it's the alpha channel)
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                #if i == 3:
                #    # alpha channel
                #    out_array[:,:,i] = numpy.ones(out_shape, dtype=dt)+channel_defaults[i]
                #    pdb.set_trace()
                #else:
                out_array[:,:,i] = numpy.zeros(out_shape, dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                doScaleRows = scale_rows_to_unit_interval
                if isColor and scale_colors_together:
                    # already scaled whole rows
                    doScaleRows = False
                out_array[:,:,i] = tile_raster_images(X[i], img_shape[0:2], tile_shape, tile_spacing, doScaleRows, False, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
            #out_array = numpy.zeros(out_shape, dtype=dt)
            out_array = numpy.ones(out_shape, dtype=dt) * 51
        else:
            out_array = numpy.ones(out_shape, dtype=dt) * .2


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval and not (isColor and scale_colors_together):
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array



def pil_imagesc(arr, epsilon = 1e-8, saveto = None):
    '''Like imagesc for Octave/Matlab, but using PIL.'''

    imarray = numpy.array(arr, dtype = numpy.float32)
    imarray -= imarray.min()
    imarray /= (imarray.max() + epsilon)
    image = Image.fromarray(imarray * 255).convert('L')
    if saveto:
        image.save(saveto)

    
