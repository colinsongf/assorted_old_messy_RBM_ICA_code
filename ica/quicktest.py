#! /usr/bin/env ipythonpl

from numpy import *
from matplotlib import pyplot
import Image
from IPython import embed

from layers import NYU2_Labeled
from util.plotting import pil_imagesc



def main():
    testNr = 0

    if testNr == 0:
        dataLayer = NYU2_Labeled({'name': 'data', 'type': 'data', 'imageSize': (480,640), 'patchSize': (10,10), 'stride': (10,10), 'colors': 4})
        patchShape = (480,640)
        patches, labels = dataLayer.getDataAndLabels(patchShape, 1, 0)

        im = pil_imagesc(reshape(patches[:,0], patchShape + (4,))[:,:,0:3])
        im.show()
        im = pil_imagesc(reshape(patches[:,0], patchShape + (4,))[:,:,3])
        im.show()
        im = pil_imagesc(reshape(patches[:,0], patchShape + (4,))[:,:,3], cmap='jet')
        im.show()
        im = pil_imagesc(reshape(labels[:,0], patchShape), cmap='jet')
        im.show()

    if testNr == 1:
        pass

    embed()



if __name__ == '__main__':
    main()
