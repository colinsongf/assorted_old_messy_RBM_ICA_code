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
from rica import RICA



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
        #self.probability = data.mean(0)         # Only for 0,1 values
        self.probability = (data > .1).mean(0)   # for -1,1 values
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



def doIndepVoxelModel(data, size, efOrder = True):
    model = IndepVoxelModel(data)
    generateIndepVoxelModel(model, data, size, efOrder)
    mutateIndepVoxelModel(model, data, size, efOrder)



def generateIndepVoxelModel(model, data, size, efOrder):
    random.seed(0)
    nShow = 50
    for ii in range(nShow):
        blob = model.generate()
        # rotate axes to x,y,z order
        if efOrder:
            blob = flat2XYZ(blob)
        else:
            blob = reshape(blob, size)

        for rr in range(1,2):
            rot = rr * 24
            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'IVMgen_%03d_rot%03d.png' % (ii, rot)))



def mutateIndepVoxelModel(model, data, size, efOrder):
    for seed in range(1):
        random.seed(seed)
        blob = model.generate()

        degreesPerFrame = 1
        framesPerMutation = 8
        for frame in range(450):
            rot = frame * degreesPerFrame

            if efOrder:
                blobPlot = flat2XYZ(blob)
            else:
                blobPlot = reshape(blob, size)
            
            plot3DShape(blobPlot, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'IVMmut_s%03d_f%03d.png' % (seed, frame)))

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



def doPCAVoxelModel(data, size, efOrder = True):
    model = PCAVoxelModel(data, dimsKeep = 30)
    
    if resman.rundir:
        pyplot.semilogy(model.pca.fracVar, 'o-')
        pyplot.title('Fractional variance in each dimension')
        pyplot.savefig(os.path.join(resman.rundir, 'fracVar.png'))
        pyplot.savefig(os.path.join(resman.rundir, 'fracVar.pdf'))
        pyplot.close()

    #pdb.set_trace()
    generatePCAVoxelModel(model, data, size, efOrder = efOrder)
    mutatePCAVoxelModel(model, data, size, mutateFn = 'mutateFewDimensions', efOrder = efOrder)
    mutatePCAVoxelModel(model, data, size, mutateFn = 'mutateMetHast', efOrder = efOrder)



def generatePCAVoxelModel(model, data, size, efOrder):
    random.seed(0)
    nShow = 45
    for ii in range(nShow):
        blob = model.generate(numDims = None)
        # rotate axes to x,y,z order
        #blob = flat2XYZ(blob, size = (2,10,20))
        #blob = flat2XYZ(blob, size = (10,10,20))
        if efOrder:
            blob = flat2XYZ(blob, size = size)
        else:
            blob = reshape(blob, size)

        #print 'blob from', blob.min(), 'to', blob.max()

        #continue
        #pdb.set_trace()

        for rr in range(24,25):
            rot = rr * 1
            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMgen_%03d_rot%03d_blocky.png' % (ii, rot)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMgen_%03d_rot%03d_smooth.png' % (ii, rot)))



def mutatePCAVoxelModel(model, data, size, mutateFn, efOrder):
    if mutateFn == 'mutateFewDimensions':
        tag = 'Few'
    elif mutateFn == 'mutateMetHast':
        tag = 'MH'
    else:
        raise Exception('unknown mutateFn: %s' % repr(mutateFn))
    
    for seed in range(5):
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
        for frame in range(720):
            rot = frame * degreesPerFrame

            if efOrder:
                blob = flat2XYZ(blob, size)
            else:
                blob = reshape(blob, size)

            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMmut%s_s%03d_f%05d_blocky.png' % (tag, seed, frame)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMmut%s_s%03d_f%05d_smooth.png' % (tag, seed, frame)))

            if mutateFn == 'mutateFewDimensions':
                if frame % framesPerMutation == 0:
                    pc, blob = model.mutateFewDimensions(pc, howmany = 1)
            else:
                pc, blob = model.mutateMetHast(pc, proposalNoise = .025)



#############################
#
# ICA Voxel Model
#
#############################

class ICAVoxelModel(object):
    '''Models voxels with ICA'''

    def __init__(self, data, dataShape, dimsKeep):
        #print datetime.now(), 'starting PCA...'
        ##self.pca = PCA(data)
        #self.pca = cached(PCA, data)
        #print datetime.now(), 'done with PCA.'
        #
        #self.dimsKeep = dimsKeep
        #
        ##pdb.set_trace()
        #
        ##dataReduced = self.pca.toWhitePC(data, numDims = self.dimsKeep)

        self.rica = RICA(imgShape = dataReduced.shape[1],
                         nFeatures = 200,
                         lambd = .0005,
                         saveDir = resman.rundir,
                         doPlots = False)
        
        #self.rica.run(dataReduced.T, plotEvery = None, maxFun = 300, whiten = False, normData = False)
        self.rica.run(data.T, plotEvery = None, maxFun = 300, whiten = False, normData = False)

        

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



def doICAVoxelModel(data, size, efOrder = True):
    random.seed(0)
    model = ICAVoxelModel(data, size, dimsKeep = 100)
    
    #if resman.rundir:
    #    pyplot.semilogy(model.pca.fracVar, 'o-')
    #    pyplot.title('Fractional variance in each dimension')
    #    pyplot.savefig(os.path.join(resman.rundir, 'fracVar.png'))
    #    pyplot.savefig(os.path.join(resman.rundir, 'fracVar.pdf'))
    #    pyplot.close()
    #
    ##pdb.set_trace()
    #generateICAVoxelModel(model, data, size, efOrder = efOrder)
    #mutateICAVoxelModel(model, data, size, mutateFn = 'mutateFewDimensions', efOrder = efOrder)
    #mutateICAVoxelModel(model, data, size, mutateFn = 'mutateMetHast', efOrder = efOrder)



def generateICAVoxelModel(model, data, size, efOrder):
    random.seed(0)
    nShow = 45
    for ii in range(nShow):
        blob = model.generate(numDims = None)
        # rotate axes to x,y,z order
        #blob = flat2XYZ(blob, size = (2,10,20))
        #blob = flat2XYZ(blob, size = (10,10,20))
        if efOrder:
            blob = flat2XYZ(blob, size = size)
        else:
            blob = reshape(blob, size)

        #print 'blob from', blob.min(), 'to', blob.max()

        #continue
        #pdb.set_trace()

        for rr in range(24,25):
            rot = rr * 1
            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMgen_%03d_rot%03d_blocky.png' % (ii, rot)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMgen_%03d_rot%03d_smooth.png' % (ii, rot)))



def mutateICAVoxelModel(model, data, size, mutateFn, efOrder):
    if mutateFn == 'mutateFewDimensions':
        tag = 'Few'
    elif mutateFn == 'mutateMetHast':
        tag = 'MH'
    else:
        raise Exception('unknown mutateFn: %s' % repr(mutateFn))
    
    for seed in range(5):
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
        for frame in range(720):
            rot = frame * degreesPerFrame

            if efOrder:
                blob = flat2XYZ(blob, size)
            else:
                blob = reshape(blob, size)

            plot3DShape(blob, smoothed = False, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMmut%s_s%03d_f%05d_blocky.png' % (tag, seed, frame)))
            plot3DShape(blob, smoothed = True, plotEdges = True, figSize = (800,800),
                        rotAngle = rot,
                        saveFilename = os.path.join(resman.rundir, 'PVMmut%s_s%03d_f%05d_smooth.png' % (tag, seed, frame)))

            if mutateFn == 'mutateFewDimensions':
                if frame % framesPerMutation == 0:
                    pc, blob = model.mutateFewDimensions(pc, howmany = 1)
            else:
                pc, blob = model.mutateMetHast(pc, proposalNoise = .025)



def main():
    useSimpleShapes = True

    if useSimpleShapes:
        labels = None
        #data = loadFromPklGz('../data/simple3DShapes/poisson_train_500.pkl.gz')
        #data = loadFromPklGz('../data/simple3DShapes/poisson_train_5000.pkl.gz')
        data = loadFromPklGz('../data/simple3DShapes/poisson_train_50000.pkl.gz'); data = data[:25000,:]
        #data = loadFromPklGz('../data/simple3DShapes/poisson_train_50000.pkl.gz')
    else:
        #labels, data = loadFromPklGz('../data/endlessforms/train_bool_50_0.pkl.gz')
        #labels, data = loadFromPklGz('../data/endlessforms/train_bool_50000_0.pkl.gz')
        labels, data = loadFromPklGz('../data/endlessforms/train_real_5000_0.pkl.gz')
        #labels, data = loadFromPklGz('../data/endlessforms/train_real_50000_0.pkl.gz')
        #labels, data = loadFromPklGz('../data/endlessforms/train_real_50_0.pkl.gz')

    size = (10,10,20)

    FAST_HACK = False
    if FAST_HACK:
        data = data[:,:400]
        size = (10,10,4)
    
    #visInput(data, labels, efOrder = not useSimpleShapes)
    #doIndepVoxelModel(data, size, efOrder = not useSimpleShapes)
    #doPCAVoxelModel(data, size, efOrder = not useSimpleShapes)
    #doICAVoxelModel(data, size, efOrder = not useSimpleShapes)



if __name__ == '__main__':
    resman.start('junk', diary = False)
    main()
    resman.stop()
