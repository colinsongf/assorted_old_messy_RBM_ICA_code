#! /usr/bin/env python

from numpy import random, vstack, zeros
import Image
from PIL import ImageFont, ImageDraw
import os, pdb

from utils import tile_raster_images, imagesc
from pca import PCA
from upsonRbm import loadUpsonData
from ResultsManager import resman

from matplotlib import pyplot



def main():
    random.seed(1)
    img_dim = 15     # 10, 15, ...
    datasets = loadUpsonData('../data/upson_rovio_1/train_%d_50000.pkl.gz' % img_dim,
                             '../data/upson_rovio_1/test_%d_50000.pkl.gz'  % img_dim)
    print 'done loading.'
    train_set_x_data, train_set_y = datasets[0]

    pca = PCA(train_set_x_data)
    print 'done PCA.'

    image = Image.fromarray(tile_raster_images(
             X = train_set_x_data,
             img_shape = (img_dim,img_dim),tile_shape = (10,10),
             tile_spacing=(1,1)))
    image.save(os.path.join(resman.rundir, 'samplesData.png'))

    pyplot.figure()
    pyplot.subplot(221); pyplot.semilogy(pca.var);  pyplot.title('pca.var')
    pyplot.subplot(222); pyplot.semilogy(pca.std);  pyplot.title('pca.std')
    pyplot.subplot(223); pyplot.semilogy(pca.fracVar);  pyplot.title('pca.fracVar')
    pyplot.subplot(224); pyplot.semilogy(pca.fracStd);  pyplot.title('pca.fracStd')
    pyplot.savefig(os.path.join(resman.rundir, 'varstd.png'))
    pyplot.close()

    #font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-lyx/cmr10.ttf', 10)
    font = ImageFont.truetype('/usr/share/texmf/fonts/opentype/public/lm/lmmono12-regular.otf', 14)

    def plotImage(xx, filename, str = None):
        arr = tile_raster_images(X = xx,
                                 img_shape = (img_dim,img_dim),tile_shape = (10,10),
                                 tile_spacing=(1,1))
        arrHeight = arr.shape[0]
        if str is not None:
            arr = vstack((arr, zeros((20, arr.shape[1]), dtype = arr.dtype)))
        image = Image.fromarray(arr)
        if str is not None:
            draw = ImageDraw.Draw(image)
            draw.text((2, arrHeight+2), str, 255, font = font)
            draw = ImageDraw.Draw(image)
        image.save(os.path.join(resman.rundir, filename))

    plotImage(pca.pc().T, 'pc.png')

    for dims in [1, 2, 5, 10, 20, 50, 100, 200, 225]:
        plotImage(pca.pcaAndBack(train_set_x_data, dims),
                  'samplesPCA_%03d.png' % dims,
                  'dims=%d' % dims)

        for ee, epsilon in enumerate([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5]):
            plotImage(pca.toZca(train_set_x_data, dims, epsilon = epsilon),
                      'samplesZCA_%03d_%02d.png' % (dims, ee),
                      'dims=%d, eps=%s' % (dims, repr(epsilon)))



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
