#! /usr/bin/env ipythonwx

import pdb
import os
import time
from numpy import *

from GitResultsManager import resman

from util.dataLoaders import loadCifarData, loadCifarDataMonochrome, loadCifarDataSubsets, loadFromPklGz
from util.endlessForms import getEFFindUrl
from util.plotting import plot3DShape, justPlotBoolArray



def flat2XYZ(blob):
    '''Reads flattened blob data in the appropriate order and creates a 3D array'''
    return transpose(reshape(blob, (10,20,10)), (2, 0, 1))



def visInput(data, labels):
    #shape = random.normal(0,1,3*3*3)
    #idx = 9         # 9 is good to check chirality... but numbers not matching up.
    #blob = reshape(data[idx,:], (20,10,10))

    nShow = max(data.shape[0], 50)
    for ii in range(nShow):
        blob = data[ii,:]
        # rotate axes to x,y,z order
        blob = flat2XYZ(blob)
        
        print ii, getEFFindUrl('http://devj.cornell.endlessforms.com', labels[ii]), '\t', sum(blob > .1)
        #print ii, '\t', sum(blob > .1)

        for rr in range(15):
            rot = rr * 24
            plot3DShape(blob, smoothed = False, plotEdges = False, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'shape_%03d_rot%03d.png' % (ii, rot)))



class IndepVoxelModel(object):
    '''Models each voxel independently'''

    def __init__(self, data):
        self.probability = data.mean(0)

    def generate(self):
        return random.rand(len(self.probability)) < self.probability



def doIndepVoxelModel(data):
    random.seed(0)
    model = IndepVoxelModel(data)
    
    nShow = 20
    for ii in range(nShow):
        blob = model.generate()
        # rotate axes to x,y,z order
        blob = flat2XYZ(blob)

        for rr in range(15):
            rot = rr * 24
            plot3DShape(blob, smoothed = True, plotEdges = False, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'IVM_%03d_rot%03d.png' % (ii, rot)))



def main():
    #labels, data = loadFromPklGz('../data/endlessforms/train_bool_50_0.pkl.gz')
    labels, data = loadFromPklGz('../data/endlessforms/train_bool_50000_0.pkl.gz')
    #labels, data = loadFromPklGz('../data/endlessforms/train_real_50_0.pkl.gz')
    
    #visInput(data, labels)
    doIndepVoxelModel(data)



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
