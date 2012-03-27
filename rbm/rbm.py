#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

import PIL, Image, pdb, sys
import copy, os, numpy, time
from numpy import *

import matplotlib
matplotlib.use('Agg') # plot with no display
from matplotlib import pyplot

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':8})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']))
rc('text', usetex=True)

import theano
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images, resman, imagesc
from logistic_sgd import load_data


#import psyco
#psyco.full()



def sigmoid(xx):
    '''Compute the logistic/sigmoid in a numerically stable way (using tanh).'''
    #return 1. / (1 + exp(-xx))
    #print 'returning', .5 * (1 + tanh(xx / 2.))
    return .5 * (1 + tanh(xx / 2.))

def logistic(xx):
    raise Exception('use sigmoid (same)')



class RBM(object):
    '''
    Implements a Restricted Boltzmann Machine. Can be with a binary or real visible layer
    '''

    def __init__(self, nVisible, nHidden, numpyRng, theanoRng, initWfactor = 1.0, visibleModel='binary'):
        '''Construct an RBM

        visibleModel = 'binary' or 'real'
        '''

        self.nVisible  = nVisible
        self.nHidden   = nHidden
        self.numpyRng  = numpyRng
        self.theanoRng = theanoRng

        self.trainIter = -1

        if visibleModel not in ('real', 'binary'):
            raise Exception('unrecognized visibleModel: %s' % repr(visibleModel))
        print 'RBM visibleModel = %s' % visibleModel

        self.realValuedVisible = (visibleModel == 'real')

        #self.v = (random.randint(0, 2, (self._sizeV, 1)) > .5) + 0
        #self.h = (random.randint(0, 2, (self._sizeH, 1)) > .5) + 0

        #self._W = random.normal(0, 1, (self._sizeH + 1, self._sizeV + 1))
        # Important: set 0,0 term to 0 (bias-bias term should not influence network)
        #self._W[0,0] = 0

        #self._reconErrorNorms = array([])

        # initialized same as RBM.py from theano
        self.W = numpy.asarray(self.numpyRng.uniform(low = -4*numpy.sqrt(6./(self.nHidden+self.nVisible)) * initWfactor,
                                                     high = 4*numpy.sqrt(6./(self.nHidden+self.nVisible)) * initWfactor,
                                                     size = (self.nVisible, self.nHidden)),
                               dtype = numpy.float32)
        self.hbias = numpy.zeros(self.nHidden,  dtype = numpy.float32)
        self.vbias = numpy.zeros(self.nVisible, dtype = numpy.float32)   # TODO change this to see if perf improves.


    #def getV(self, withBias = False):
    #    '''Visible nodes'''
    #    if withBias:
    #        return self._v[0:]
    #    else:
    #        return self._v[1:]
    #
    #def setV(self, val):
    #    assert (len(val) == self._sizeV)
    #    self._v = array([hstack((array([1]), val.squeeze()))]).T
    #
    #v = property(getV, setV)
    #
    #
    #def getH(self, withBias = False):
    #    '''Hidden nodes'''
    #    if withBias:
    #        return self._h[0:]
    #    else:
    #        return self._h[1:]
    #
    #def setH(self, val):
    #    assert (len(val) == self._sizeH)
    #    self._h = array([hstack((array([1]), val.squeeze()))]).T
    #
    #h = property(getH, setH)


    #def getW(self):
    #    '''Combined weight and bias matrix'''
    #    return self._W
    #
    #def setW(self, val):
    #    assert (val.shape[0] == self._sizeH + 1)
    #    assert (val.shape[1] == self._sizeV + 1)
    #    self._W = val
    #    self._W[0,0] = 0
    #
    #W = property(getW, setW)


    def getReconErrorNorms(self):
        '''Vector of reconstruction error norms for each training vector seen so far.'''
        return self._reconErrorNorms

    reconErrorNorms = property(getReconErrorNorms)


    def energy(self):
        '''Compute energy of the network given the visible states,
        hidden states, and current extended weight matrix.

        #vv and hh must be column vectors.

        #To account for biases, vv[0,0] and hh[0,0] must be 1, and WW[0,0]
        #must be 0.'''

        assert (self._v[0,0] == 1)
        assert (self._h[0,0] == 1)
        assert (self._W[0,0] == 0)

        return -dot(dot(self._W, self._v).T, self._h)


    def v2h(self):
        '''Do a visible to hidden step.'''
        self._h = dot(self._W, self._v)
        self._h = (logistic(self._h) > random.uniform(0, 1, self._h.shape)) + 0
        self._h[0,0] = 1    # Bias term


    def h2v(self, activation = 'logisticBinary', param = 1, returnNoisefree = False):
        '''Do a hidden to visible step.'''
        ret = None
        self._v = dot(self._W.T, self._h)
        if activation == 'logisticBinary':
            self._v = (logistic(self._v) > random.uniform(0, 1, self._v.shape)) + 0
        elif activation == 'gaussianReal':
            if returnNoisefree:
                ret = copy.copy(self._v)
            self._v += param * random.normal(0, 1, self._v.shape)
        else:
            self._v[0,0] = 1    # Bias term
            raise Exception('Unknown activation: %s' % activation)
        self._v[0,0] = 1    # Bias term

        return ret


    def learn1(self, vv, epsilon = .1, activationH2V = 'logisticBinary', param = 1):
        '''Shift weights to better reconstruct the given visible vector.'''

        self.v = vv
        self.v2h()
        vihjData  = dot(self._h, self._v.T)
        if activationH2V == 'gaussianReal':
            noisefreeV = self.h2v(activation = activationH2V,
                                  param      = param,
                                  returnNoisefree = True)
            self._reconErrorNorms = hstack((self._reconErrorNorms,
                                            linalg.norm(noisefreeV[1:].squeeze() - vv)))
        else:
            self.h2v(activation = activationH2V,
                     param      = param)
            self._reconErrorNorms = hstack((self._reconErrorNorms,
                                            linalg.norm(self._v[1:].squeeze() - vv)))
        self.v2h()
        vihjRecon = dot(self._h, self._v.T)
        self._W += epsilon * (vihjData - vihjRecon)

        #print 'self._W[0,0] was', self._W[0,0]
        self._W[0,0] = 0
        

    def reconErrorVec(self, vv, activationH2V = 'logisticBinary'):
        '''Performs a V2H step, an H2V step, and then reports the
        reconstruction error as a vector of differences.'''

        self.v = vv
        self.v2h()
        self.h2v(activation = activationH2V)
        return self._v[1:].squeeze() - vv


    def reconErrorNorm(self, vv, activationH2V = 'logisticBinary'):
        '''2 Norm of the reconErrorVec for vector vv.'''

        return linalg.norm(self.reconErrorVec(vv, activationH2V))


    def plot(self, nSubplots = 3, skipH = False, skipW = False, skipV = False):
        '''Plot the hidden layer, weight layer, and visible layers in
        the current figure.'''
        lclr = [1,.47,0]

        curSubplot = 1
        if not skipH:
            ax = pylab.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            pylab.imshow(self._h.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeH < 25:
                pylab.xticks(arange(self._h.shape[0])-.5)
                pylab.yticks(arange(self._h.shape[1])-.5)
            else:
                pylab.xticks([])
                pylab.yticks([])
            pylab.axvline(.5, color=lclr, linewidth=2)

        if not skipW:
            ax = pylab.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            pylab.imshow(self._W, cmap='gray', interpolation='nearest', vmin=-2, vmax=2)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeH < 25 and self._sizeV < 25:
                pylab.xticks(arange(self._W.shape[1])-.5)
                pylab.yticks(arange(self._W.shape[0])-.5)
            else:
                pylab.xticks([])
                pylab.yticks([])
            pylab.axvline(.5, color=lclr, linewidth=2)
            pylab.axhline(.5, color=lclr, linewidth=2)

        if not skipV:
            ax = pylab.subplot(nSubplots,1,curSubplot)
            curSubplot += 1
            pylab.imshow(self._v.T, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if self._sizeV < 25:
                pylab.xticks(arange(self._v.shape[0])-.5)
                pylab.yticks(arange(self._v.shape[1])-.5)
            else:
                pylab.xticks([])
                pylab.yticks([])
            pylab.axvline(.5, color=lclr, linewidth=2)




    def plotWeightHist(self, values):
        pyplot.hist(values);
        pyplot.title('mm = %g' % mean(fabs(values)))
        ax = pyplot.gca()
        ax.xaxis.set_major_locator(pyplot.MaxNLocator(nbins=6, steps=[1,2,5,10]))

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)),axis = 1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        ''' This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost function)
        '''
        pre_sigmoid_activation = dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theanoRng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX

        #h1_sample = self.theanoRng.binomial(size = h1_mean.shape, n = 1, p = h1_mean,
        #        dtype = theano.config.floatX)

        h1_sample = random.uniform(size=h1_mean.shape) < h1_mean

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost function)
        '''
        if self.realValuedVisible:
            activation = dot(hid, self.W.T) + self.vbias
            return activation
        else:
            pre_sigmoid_activation = dot(hid, self.W.T) + self.vbias
            return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample, noiseSigma = 1):
        ''' This function infers state of visible units given hidden units.'''
        if self.realValuedVisible:
            v1_mean = self.propdown(h0_sample)  # Real valued version
            # get a sample of the visible given their activation
            v1_sample = v1_mean + random.normal(loc=0, scale=noiseSigma, size=v1_mean.shape)  # Real valued version

            return [None, v1_mean, v1_sample]

        else:
            # compute the activation of the visible given the hidden sample
            pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
            # get a sample of the visible given their activation
            v1_sample = random.uniform(size=v1_mean.shape) < v1_mean

            return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]



    # Old def get_cost_updates(self, train_x, lr = 0.1, persistent=None, k =1):
    def train(self, train_x, lr = 0.1, persistent = None, k = 1, metrics = False, plotWeights = False):
        '''
        This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size, number of
        hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.
        '''

        self.trainIter += 1

        batch_size = train_x.shape[0]

        # compute positive phase
        ph1_pre_sigmoid, ph1_mean, h1_sample = self.sample_h_given_v(train_x)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = h1_sample
        else:
            raise Exception('not implemented yet')
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        #[pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates = \
        #    theano.scan(self.gibbs_hvh,
        #            # the None are place holders, saying that
        #            # chain_start is the initial state corresponding to the
        #            # 6th output
        #            outputs_info = [None, None, None,None,None,chain_start],
        #            n_steps = k)

        pv2_pre_sigmoid, pv2_mean, v2_sample = self.sample_v_given_h(h1_sample)
        ph2_pre_sigmoid, ph2_mean, h2_sample = self.sample_h_given_v(v2_sample)

        # [JBY] This is currently implemented with the "safe, slow"
        # parameters: positive and negative terms are computed with
        # the sampled values, except for the final h, where
        # probabilities are used.


        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        #chain_end = nv_samples[-1]

        #cost = mean(self.free_energy(self.input)) - mean(self.free_energy(chain_end))

        # We must not compute the gradient through the gibbs sampling
        #gparams = T.grad(cost, self.params,consider_constant = [chain_end])

        # update parameters with average over mini-batch
        if not plotWeights:
            self.vbias += lr * mean(v2_sample - train_x, 0)
            self.hbias += lr * mean(h1_sample - ph2_mean, 0)
            self.W     += (lr / batch_size) * (dot(train_x.T, h1_sample) - dot(v2_sample.T, ph2_mean))
        else:
            dvbias = lr * mean(v2_sample - train_x, 0)
            dhbias = lr * mean(h1_sample - ph2_mean, 0)
            dW     = (lr / batch_size) * (dot(train_x.T, h1_sample) - dot(v2_sample.T, ph2_mean))

            pyplot.figure()
            pyplot.subplot(231); self.plotWeightHist(self.vbias)
            pyplot.subplot(232); self.plotWeightHist(self.W.flatten())
            pyplot.subplot(233); self.plotWeightHist(self.hbias)
            pyplot.subplot(234); self.plotWeightHist(dvbias)
            pyplot.subplot(235); self.plotWeightHist(dW.flatten())
            pyplot.subplot(236); self.plotWeightHist(dhbias)
            pyplot.savefig(os.path.join(resman.rundir, 'weightHist_%s' % plotWeights))
            pyplot.close()
            
            #pyplot.figure()
            #pyplot.imshow(ph1_mean, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            #pyplot.savefig(os.path.join(resman.rundir, 'hiddenProb_%05d' % self.trainIter))
            #pyplot.close()

            image = Image.fromarray(ph1_mean * 256)
            image.convert('L').save(os.path.join(resman.rundir, 'hiddenProb_%s.png' % plotWeights))

            self.vbias += dvbias
            self.hbias += dhbias
            self.W     += dW
            
        if not metrics:
            return

        # constructs the update dictionary
        #for gparam, param in zip(gparams, self.params):
        #    # make sure that the learning rate is of the right dtype
        #    updates[param] = param - gparam * T.cast(lr, dtype = theano.config.floatX)
        if persistent:
            neverGetHere
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_costs = self.get_reconstruction_cost(train_x, pv2_pre_sigmoid, pv2_mean)

        #if self.lastParams is not None:
        #    self.paramDelta = self.params - self.lastParams
        #updates[self.paramDelta] = self.paramDelta
        #self.lastParams = self.params

        return monitoring_costs

    def get_pseudo_likelihood_cost(self, updates):
        '''Stochastic approximation to the pseudo-likelihood'''
        raise Exception('probably do not call this')

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name = 'bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:,bit_i_idx], 1-xi[:,bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, train_x, pv_pre_sigmoid, pv_mean):
        '''Now returns approximation to reconstruction error and actual squared error (per pixel)

        Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as input. To
        understand why this is so you need to understand a bit about how
        Theano works. Whenever you compile a Theano function, the computational
        graph that you pass as input gets optimized for speed and stability. This
        is done by changing several parts of the subgraphs with others. One
        such optimization expresses terms of the form log(sigmoid(x)) in terms of softplus.
        We need this optimization for the cross-entropy since sigmoid of
        numbers larger than 30. (or even less then that) turn to 1. and numbers
        smaller than  -30. turn to 0 which in terms will force theano
        to compute log(0) and therefore we will get either -inf or NaN
        as cost. If the value is expressed in terms of softplus we do
        not get this undesirable behaviour. This optimization usually works
        fine, but here we have a special case. The sigmoid is applied inside
        the scan op, while the log is outside. Therefore Theano will only
        see log(scan(..)) instead of log(sigmoid(..)) and will not apply
        the wanted optimization. We can not go and replace the sigmoid
        in scan with something else also, because this only needs to be
        done on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of scan,
        and apply both the log and sigmoid outside scan such that Theano
        can catch and optimize the expression.
        '''

        #cross_entropy = T.mean(
        #        T.sum(self.input*T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
        #        (1 - self.input)*T.log(1-T.nnet.sigmoid(pre_sigmoid_nv)), axis = 1))

        # Use log1p for greater numerical stability. Still not perfect though.
        # log(sigmoid(-30)) = -29.999
        # log(sigmoid(-40)) = -inf
        # -log1p(exp(30))   = -30.000000000000092
        # -log1p(exp(705))  = -705.0
        # -log1p(exp(710))  = -inf

        if self.realValuedVisible:
            # Skip this for real-valued neurons until a new metric is implemented
            cross_entropy = 0.0
        else:
            cross_entropy = mean(sum(-train_x*numpy.log1p(exp(-pv_pre_sigmoid)) - (1-train_x)*numpy.log1p(exp(pv_pre_sigmoid)), 1))

        recon_err = mean((train_x - pv_mean)**2)

        return cross_entropy, recon_err



def oldmain():
    Nv = 20
    Nh = 20
    
    rbm = RBM(Nv, Nh)

    #pylab.figure(1)

    energies = array([[]])
    for ii in range(100):
        ee = rbm.energy()
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))

        if mod(ii, 5) == 0:
            pylab.clf()
            rbm.plot(nSubplots = 4)
            ax = pylab.subplot(4,1,4)
            pylab.plot(energies[0])
            pylab.show()
            time.sleep(.1)

        rbm.v2h()
        print '  v->h step'

        ee = rbm.energy()
        print '%d Energy is' % ii, ee
        energies = hstack((energies, ee))
        
        if mod(ii, 5) == 999:
            pylab.clf()
            rbm.plot(nSubplots = 4)
            ax = pylab.subplot(4,1,4)
            pylab.plot(energies[0])
            pylab.show()
            time.sleep(.1)

        rbm.h2v()
        print '  h->v step'

    def __str__(self):
        return 'RBM(nVisible = %d, nHidden = %d, trainIter = %d)' % (self.nVisible, self.nHidden, self.trainIter)



def test_rbm(learning_rate=0.1, training_epochs = 15,
             datasets = None, batch_size = 20,
             n_chains = 20, n_samples = 14, output_folder = 'rbm_plots',
             img_dim = 28, n_input = None, n_hidden = 500, quickHack = False,
             visibleModel = 'binary'):
    '''
    Demonstrate how to train an RBM.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    '''

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x,  test_set_y  = datasets[2]

    if quickHack:
        train_set_x = train_set_x[:2500,:]
        train_set_y = train_set_y[:2500]

    if n_input == None:
        n_input = img_dim ** 2

    print ('(%d, %d, %d) %d dimensional examples in (train, valid, test)' % 
           (train_set_x.shape[0], valid_set_x.shape[0], test_set_x.shape[0], train_set_x.shape[1]))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size

    print 'n_train_batches is', n_train_batches



    # allocate symbolic variables for the data
    #index = T.lscalar()    # index to a [mini]batch
    #x     = T.matrix('x')  # the data is presented as rasterized images


    rng        = numpy.random.RandomState(1)
    theanoRng  = RandomStreams( rng.randint(2**30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    #persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),dtype=theano.config.floatX))

    # construct the RBM class
    rbm = RBM(nVisible=n_input, \
              nHidden = n_hidden, numpyRng = rng, theanoRng = theanoRng, visibleModel = visibleModel)



    # get the cost and the gradient corresponding to one step of CD-15
    #cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k = 15)


    #################################
    #     Training the RBM          #
    #################################

    print 'starting training.'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)


    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    #train_rbm = theano.function([index], cost,
    #       updates = updates,
    #       givens = { x: train_set_x[index*batch_size:(index+1)*batch_size]},
    #       name = 'train_rbm')

    plotting_time = 0.
    start_time = time.clock()


    # go through training epochs
    meanCosts = []
    ii = -1
    metrics = array([])
    plotEvery = 100
    for epoch in xrange(training_epochs):
        # go through the training set
        for batch_index in xrange(n_train_batches):
            #print 'about to train using exemplars %d to %d.' % (batch_index*batch_size, (batch_index+1)*batch_size)

            ii += 1
            if ii % plotEvery == 0:
                plotWeights = '%03i_%05i' % (epoch, batch_index)
                calcMetrics = True
            else:
                plotWeights = False
                calcMetrics = False

            # metric is xEntropyCost, reconError
            metric = rbm.train(train_set_x[batch_index*batch_size:(batch_index+1)*batch_size],
                               lr = learning_rate, metrics = calcMetrics, plotWeights = plotWeights)

            if calcMetrics:
                if len(metrics) == 0:
                    metrics = array([metric])
                else:
                    metrics = vstack((metrics, metric))

            if ii % plotEvery == 0:
                # Plot filters after each single step
                plotting_start = time.clock()
                # Construct image from the weight matrix
                image = Image.fromarray(tile_raster_images(
                         X = rbm.W.T,
                         img_shape = (img_dim,img_dim),tile_shape = (10,10),
                         tile_spacing=(1,1)))
                image.save(os.path.join(output_folder, 'filters_at_epoch_batch_%03i_%05i.png' % (epoch, batch_index)))
                plotting_stop = time.clock()
                plotting_time += (plotting_stop - plotting_start)

                #print '  Training epoch %d batch %d, xEntropyCost is ' % (epoch, batch_index), numpy.mean(mean_cost),
                print '  Training epoch %d batch %d, xEntropyCost is ' % (epoch, batch_index), metrics[-1,0],
                print '\trecon error ', metrics[-1,1]

        thisEpochStart =  epoch   *n_train_batches/plotEvery
        thisEpochEnd   = (epoch+1)*n_train_batches/plotEvery
        epochMeanXEnt  = mean(metrics[thisEpochStart:thisEpochEnd,0])
        epochMeanRecon = mean(metrics[thisEpochStart:thisEpochEnd,1])
        print 'Training epoch %d mean xEntropyCost is ' % (epoch), epochMeanXEnt, '\trecon error ', epochMeanRecon

        meanCosts.append(epochMeanXEnt)

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(tile_raster_images(
                 X = rbm.W.T,
                 img_shape = (img_dim,img_dim),tile_shape = (10,10),
                 tile_spacing=(1,1)))
        image.save(os.path.join(output_folder, 'filters_at_epoch_%03i.png' % epoch))
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    # MERGE: this might fail?? (later: not sure why)
    plotting_start = time.clock()
    pyplot.plot(metrics)
    pyplot.savefig(os.path.join(output_folder, 'reconErr.png'))
    plotting_time += (time.clock() - plotting_start)
    
    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' %(pretraining_time/60.))
    print ('Plotting took %f minutes' %(plotting_time/60.))


    #################################
    #     Sampling from the RBM     #
    #################################


    # find out the number of test samples
    number_of_test_samples = test_set_x.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    #test_idx = rng.randint(number_of_test_samples-n_chains)
    #persistent_vis_chain = theano.shared(numpy.asarray(
    #        test_set_x.get_value(borrow=True)[test_idx:test_idx+n_chains],
    #        dtype=theano.config.floatX))

    plot_every = 1
    # define one step of Gibbs sampling (mf = mean-field)
    # define a function that does `plot_every` steps before returning the sample for
    # plotting
    #[presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples], updates =  \
    #                    theano.scan(rbm.gibbs_vhv,
    #                            outputs_info = [None, None,None,None,None,persistent_vis_chain],
    #                            n_steps = plot_every)


    # add to updates the shared variable that takes care of our persistent
    # chain :.
    #updates.update({ persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    #sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
    #                            updates = updates,
    #                            name = 'sample_fn')


    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(((img_dim+1)*n_samples+1,(img_dim+1)*n_chains-1), dtype='uint8')
    for ii in xrange(n_chains):
        # generate `plot_every` intermediate samples that we discard, because successive samples in the chain are too correlated
        test_idx = rng.randint(number_of_test_samples)
        
        samples = numpy.zeros((n_chains, img_dim*img_dim))

        visMean = test_set_x[test_idx,:]
        visSample = visMean
        for jj in xrange(n_samples):
            samples[jj,:] = visMean # show the mean, but use the sample for gibbs steps
            if jj == n_samples-1: break  # skip the last for speed
            plot_every = 2**jj  # exponentially increasing number of gibbs samples. max for n_samples=14 is 2^12
            for ss in xrange(plot_every):
                visMean, visSample = rbm.gibbs_vhv(visSample)[4:6]   # 4 for mean, 5 for sample

        print ' ... plotting sample ', ii
        image_data[1:-1,(img_dim+1)*ii:(img_dim+1)*ii+img_dim] = tile_raster_images(
                X = samples,
                img_shape = (img_dim,img_dim),
                tile_shape = (n_samples, 1),
                tile_spacing = (1,1))
        #image_data[(img_dim+1)*idx:(img_dim+1)*idx+img_dim,:] = tile_raster_images(
        #        X = samples,
        #        img_shape = (img_dim,img_dim),
        #        tile_shape = (1, n_chains),
        #        tile_spacing = (1,1))
        # construct image

    image = Image.fromarray(image_data)
    image.save(os.path.join(output_folder, 'samples.png'))
    #os.chdir('../')
    return meanCosts



if __name__ == '__main__':
    resman.start('junk', diary = True)
    datasets = load_data('../data/mnist.pkl.gz', shared = False)
    print 'done loading.'
    test_rbm(datasets = datasets,
             training_epochs = 1,
             n_hidden = 500,
             learning_rate = .002,
             output_folder = resman.rundir,
             quickHack = False)
    resman.stop()
