#! /usr/bin/env ipythonpl

import pdb
import os
import time
from numpy import *
from datetime import datetime
import scipy.stats

from GitResultsManager import resman

from matplotlib import pyplot

from util.dataLoaders import loadCifarData, loadCifarDataMonochrome, loadCifarDataSubsets, loadFromPklGz
from util.endlessForms import getEFFindUrl
from util.plotting import plot3DShape, justPlotBoolArray
from util.cache import memoize, cached
from rbm.pca import PCA



def flat2XYZ(blob, size=(10,10,20)):
    '''Reads flattened blob data in the appropriate order and creates a 3D array'''
    if size == (10,10,20):
        return transpose(reshape(blob, (10,20,10)), (2, 0, 1))
    elif size == (1,10,20):
        return transpose(reshape(blob, (1,20,10)), (2, 0, 1))
    elif size == (2,10,20):
        return transpose(reshape(blob, (2,20,10)), (2, 0, 1))
    else:
        raise Exception('unhandled size: %s' % repr(size))



#def XYZ2Flat(blob):
#    '''Reads data in a 3D array and creates flattened blob data in the appropriate order'''
#    return transpose(reshape(blob, (10,20,10)), (2, 0, 1))



#############################
#
# Visualize Input
#
#############################

def visInput(data, labels = None, efOrder = True):
    #shape = random.normal(0,1,3*3*3)
    #idx = 9         # 9 is good to check chirality... but numbers not matching up.
    #blob = reshape(data[idx,:], (20,10,10))

    nShow = min(data.shape[0], 50)
    for ii in range(nShow):
        blob = data[ii,:]
        # rotate axes to x,y,z order
        if efOrder:
            blob = flat2XYZ(blob)
        else:
            blob = reshape(blob, (10,10,-1))

        if labels:
            print ii, getEFFindUrl('http://devj.cornell.endlessforms.com', labels[ii]), '\t', sum(blob > .1)
            #print ii, '\t', sum(blob > .1)

        for rr in range(1,2):
            rot = rr * 24
            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'shape_%03d_rot%03d.png' % (ii, rot)))



#############################
#
# Independent Voxel Model
#
#############################

class IndepVoxelModel(object):
    '''Models each voxel independently'''

    def __init__(self, data):
        self.probability = data.mean(0)
        self.dim = len(self.probability)
        
    def generate(self):
        return random.rand(self.dim) < self.probability

    def nearby(self, data, howmany = .1):
        idxMutate = random.choice(self.dim, int(howmany*self.dim), replace=False)
        nMutate = len(idxMutate)
        if nMutate == 0:
            print 'WARNING: degenerate mutation'
        data[idxMutate] = random.rand(nMutate) < self.probability[idxMutate]
        return data



def doIndepVoxelModel(data):
    model = IndepVoxelModel(data)
    #generateIndepVoxelModel(model, data)
    mutateIndepVoxelModel(model, data)



def generateIndepVoxelModel(model, data):
    random.seed(0)
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



def mutateIndepVoxelModel(model, data):
    for seed in range(1):
        random.seed(seed)
        blob = model.generate()

        degreesPerFrame = 1
        framesPerMutation = 8
        for frame in range(450):
            rot = frame * degreesPerFrame

            plot3DShape(flat2XYZ(blob), smoothed = False, plotEdges = False, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'IVM_s%03d_f%03d.png' % (seed, frame)))

            if frame % framesPerMutation == 0:
                blob = model.nearby(blob)



#############################
#
# PCA Voxel Model
#
#############################

class PCAVoxelModel(object):
    '''Models voxels with PCA'''

    def __init__(self, data, dimsKeep):
        print datetime.now(), 'starting PCA...'
        #self.pca = PCA(data)
        self.pca = cached(PCA, data)
        print datetime.now(), 'done with PCA.'

        self.dimsKeep = dimsKeep

        #self.asPc = self.pca.toPC(data, numDims = self.dimsKeep)
        #self.asPcW = self.pca.toWhitePC(data, numDims = self.dimsKeep)

        #print datetime.now(), 'starting white projection'
        #dataWhite = self.pca.toZca(data, epsilon = 1e-6)
        #print datetime.now(), 'done with white projection'

        #self.probability = data.mean(0)
        #self.dim = len(self.probability)

        
    def generate(self, numDims = None, fullReturn = False):
        if numDims is None:
            numDims = self.dimsKeep
        genPc = random.randn(1,numDims)

        ret = self.pca.fromWhitePC(genPc)

        if fullReturn:
            return genPc, ret
        else:
            return ret


    def mutateFewDimensions(self, pc, howmany = 3):
        assert(howmany <= self.dimsKeep)
        assert(prod(pc.shape) == self.dimsKeep)

        idxMutate = random.choice(self.dimsKeep, howmany, replace=False)
        nMutate = len(idxMutate)
        if nMutate == 0:
            print 'WARNING: degenerate mutation'

        newPc = pc[:]
        newPc[0,idxMutate] = random.randn(nMutate)

        return newPc, self.pca.fromWhitePC(newPc)


    def mutateRandomWalk(self, pc, oldDelta = None, noise = .1):
        # Use mutateMetHast instead
        raise Exception
    
        assert(prod(pc.shape) == self.dimsKeep)

        if oldDelta is None:
            oldDelta = zeros(self.dimsKeep)

        delta = oldDelta + random.randn(self.dimsKeep) * noise
        newPc = pc + delta

        return newPc, delta, self.pca.fromWhitePC(newPc)


    def mutateMetHast(self, pc, proposalNoise = .1):
        assert(prod(pc.shape) == self.dimsKeep)

        proposedPc = pc + random.randn(self.dimsKeep) * proposalNoise

        # PC are whitened so cov is identity
        pCurrent  = prod(scipy.stats.norm.pdf(pc))
        pProposed = prod(scipy.stats.norm.pdf(proposedPc))

        ratio = pProposed / pCurrent

        if ratio > 1 or random.rand() < ratio:
            print '     accept'
            pc = proposedPc
        else:
            print '            reject'

        return pc, self.pca.fromWhitePC(pc)



def doPCAVoxelModel(data):
    model = PCAVoxelModel(data, dimsKeep = 15)
    
    if resman.rundir:
        pyplot.semilogy(model.pca.fracVar, 'o-')
        pyplot.title('Fractional variance in each dimension')
        pyplot.savefig(os.path.join(resman.rundir, 'fracVar.png'))
        pyplot.savefig(os.path.join(resman.rundir, 'fracVar.pdf'))
        pyplot.close()

    #pdb.set_trace()
    generatePCAVoxelModel(model, data)
    #mutatePCAVoxelModel(model, data, mutateFn = 'mutateFewDimensions')
    #mutatePCAVoxelModel(model, data, mutateFn = 'mutateMetHast')



def generatePCAVoxelModel(model, data):
    random.seed(0)
    nShow = 45
    for ii in range(nShow):
        blob = model.generate(numDims = None)
        # rotate axes to x,y,z order
        #blob = flat2XYZ(blob, size = (2,10,20))
        blob = flat2XYZ(blob, size = (10,10,20))

        #print 'blob from', blob.min(), 'to', blob.max()

        #continue
        #pdb.set_trace()

        for rr in range(180):
            rot = rr * 1
            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVM_%03d_rot%03d_blocky.png' % (ii, rot)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVM_%03d_rot%03d_smooth.png' % (ii, rot)))



def mutatePCAVoxelModel(model, data, mutateFn):
    for seed in range(10):
        random.seed(seed)
        blob = 0
        tries = 100
        while sum(blob > .1) == 0 and tries > 0:
            pc, blob = model.generate(fullReturn = True)
            tries -= 1

        #if mutateFn == 'mutateMetHast':
        #    delta = None  # no delta yet

        #pdb.set_trace()

        degreesPerFrame = 1
        framesPerMutation = 1
        for frame in range(2000):
            rot = frame * degreesPerFrame

            #blob = flat2XYZ(blob, size = (2,10,20))
            blob = flat2XYZ(blob, size = (10,10,20))

            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVM_s%03d_f%05d_blocky.png' % (seed, frame)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVM_s%03d_f%05d_smooth.png' % (seed, frame)))

            if mutateFn == 'mutateFewDimensions':
                if frame % framesPerMutation == 0:
                    pc, blob = model.mutateFewDimensions(pc, howmany = 3)
            elif mutateFn == 'mutateMetHast':
                pc, blob = model.mutateMetHast(pc, proposalNoise = .025)
            else:
                raise Exception('unknown mutateFn: %s' % repr(mutateFn))
                    



def main():
    useSimpleShapes = True

    if useSimpleShapes:
        labels = None
        data = loadFromPklGz('../data/simple3DShapes/poisson_train_5000.pkl.gz')
        #data = loadFromPklGz('../data/simple3DShapes/poisson_train_50000.pkl.gz')
    else:
        #labels, data = loadFromPklGz('../data/endlessforms/train_bool_50_0.pkl.gz')
        #labels, data = loadFromPklGz('../data/endlessforms/train_bool_50000_0.pkl.gz')
        labels, data = loadFromPklGz('../data/endlessforms/train_real_50000_0.pkl.gz')
        #labels, data = loadFromPklGz('../data/endlessforms/train_real_50_0.pkl.gz')


    FAST_HACK = False
    if FAST_HACK:
        data = data[:,:400]

    visInput(data, labels, efOrder = not useSimpleShapes)
    #doIndepVoxelModel(data)
    #doPCAVoxelModel(data)



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
