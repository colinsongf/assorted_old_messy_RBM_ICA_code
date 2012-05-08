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
rc('font',**{'size':8})

# WARNING: The next two lines are *very slow*! Use only if necessary
# for making very pretty plots.
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':8})
#rc('text', usetex=True)

from utils import tile_raster_images, imagesc, load_mnist_data, saveToFile
from ResultsManager import resman
from pca import PCA



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

    def __init__(self, nVisible, nHidden, numpyRng, initWfactor = 1.0, visibleModel='binary'):
        '''Construct an RBM

        visibleModel = 'binary' or 'real'
        '''

        self.nVisible  = nVisible
        self.nHidden   = nHidden
        self.numpyRng  = numpyRng

        self.trainIter = -1

        if visibleModel not in ('real', 'binary'):
            raise Exception('unrecognized visibleModel: %s' % repr(visibleModel))
        print 'RBM visibleModel = %s' % visibleModel
        self.realValuedVisible = (visibleModel == 'real')

        self.W = numpy.asarray(self.numpyRng.uniform(low = -4*numpy.sqrt(6./(self.nHidden+self.nVisible)) * initWfactor,
                                                     high = 4*numpy.sqrt(6./(self.nHidden+self.nVisible)) * initWfactor,
                                                     size = (self.nVisible, self.nHidden)),
                               dtype = numpy.float32)
        self.hbias = numpy.zeros(self.nHidden,  dtype = numpy.float32)
        self.vbias = numpy.zeros(self.nVisible, dtype = numpy.float32)   # TODO change this to see if perf improves.



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
        the hidden units.
        '''
        pre_sigmoid_activation = dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]


    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)

        # get a sample of the hiddens given their activation    
        h1_sample = random.uniform(size=h1_mean.shape) < h1_mean

        return [pre_sigmoid_h1, h1_mean, h1_sample]


    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units.
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


    def train(self, train_x, lr = 0.1, persistent = None, k = 1, metrics = False,
              plotWeights = False, output_dir = 'rbm_plots'):
        '''
        This functions implements one step of CD-k or PCD-k (no PCD yet)

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size, number of
        hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns the costs.
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

        pv2_pre_sigmoid, pv2_mean, v2_sample = self.sample_v_given_h(h1_sample)
        ph2_pre_sigmoid, ph2_mean, h2_sample = self.sample_h_given_v(v2_sample)


        # [JBY] This is currently implemented with the "safe, slow"
        # parameters: positive and negative terms are computed with
        # the sampled values, except for the final h, where
        # probabilities are used.

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
            pyplot.savefig(os.path.join(output_dir, 'weightHist_%s' % plotWeights))
            pyplot.close()
            
            #pyplot.figure()
            #pyplot.imshow(ph1_mean, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            #pyplot.savefig(os.path.join(output_dir, 'hiddenProb_%05d' % self.trainIter))
            #pyplot.close()

            image = Image.fromarray(ph1_mean * 256)
            image.convert('L').save(os.path.join(output_dir, 'hiddenProb_%s.png' % plotWeights))

            self.vbias += dvbias
            self.hbias += dhbias
            self.W     += dW
            
        if not metrics:
            return

        if persistent:
            raise Exception('not implemented yet')
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_costs = self.get_reconstruction_cost(train_x, pv2_pre_sigmoid, pv2_mean)

        return monitoring_costs

    def get_pseudo_likelihood_cost(self, updates):
        '''Stochastic approximation to the pseudo-likelihood'''
        raise Exception('probably do not call this')

        import theano
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
        '''Now returns approximation to reconstruction error and
        actual squared error (per pixel)
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



def test_rbm(learning_rate=0.1, training_epochs = 15,
             datasets = None, batch_size = 20,
             n_chains = 20, n_samples = 14, output_dir = 'rbm_plots',
             img_dim = 28, n_input = None, n_hidden = 500, quickHack = False,
             visibleModel = 'binary', initWfactor = 1.0,
             imgPlotFunction = None):
    '''
    Demonstrate how to train an RBM.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain


    :param visibleModel: 'real' or 'binary'

    :param initWfactor: Typicaly 1 for binary or .01 for real

    XXX:param pcaDims: None to skip PCA or >0 to use PCA to reduce dimensionality of data first.

    '''

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x,  test_set_y  = datasets[2]

    if quickHack:
        train_set_x = train_set_x[:2500,:]
        if train_set_y is not None:
            train_set_y = train_set_y[:2500]

    print ('(%d, %d, %d) %d dimensional examples in (train, valid, test)' % 
           (train_set_x.shape[0], valid_set_x.shape[0], test_set_x.shape[0], train_set_x.shape[1]))

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    print 'n_train_batches is', n_train_batches

    rng        = numpy.random.RandomState(1)

    if n_input is None:
        n_input = train_set_x.shape[1]

    # construct the RBM class
    rbm = RBM(nVisible=n_input, nHidden = n_hidden, numpyRng = rng,
              visibleModel = visibleModel, initWfactor = initWfactor)


    #################################
    #     Training the RBM          #
    #################################

    print 'starting training.'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

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
                               lr = learning_rate, metrics = calcMetrics, plotWeights = plotWeights,
                               output_dir = output_dir)

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
                         X = imgPlotFunction(rbm.W.T) if imgPlotFunction else rbm.W.T,
                         img_shape = (img_dim,img_dim),tile_shape = (10,10),
                         tile_spacing=(1,1)))
                image.save(os.path.join(output_dir, 'filters_at_epoch_batch_%03i_%05i.png' % (epoch, batch_index)))
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
                 X = imgPlotFunction(rbm.W.T) if imgPlotFunction else rbm.W.T,
                 img_shape = (img_dim,img_dim),tile_shape = (10,10),
                 tile_spacing=(1,1)))
        image.save(os.path.join(output_dir, 'filters_at_epoch_%03i.png' % epoch))
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    plotting_start = time.clock()
    pyplot.plot(metrics)
    pyplot.savefig(os.path.join(output_dir, 'reconErr.png'))
    plotting_time += (time.clock() - plotting_start)
    
    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' %(pretraining_time/60.))
    print ('Plotting took %f minutes' %(plotting_time/60.))


    #################################
    #   Plot some samples from RBM  #
    #################################


    # find out the number of test samples
    number_of_test_samples = test_set_x.shape[0]

    plot_every = 1

    # if imgPlotFunction is defined, then also plot before function if
    # the data is of the same dimension (e.g. for ZCA, but not for
    # PCA).
    plotRawAlso = (imgPlotFunction and train_set_x.shape[0] == img_dim * img_dim)
        
    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.ones(((img_dim+1)*n_samples-1,(img_dim+1)*n_chains-1), dtype='uint8') * 51  # dark gray
    if plotRawAlso:
        image_data_raw = numpy.ones(((img_dim+1)*n_samples-1,(img_dim+1)*n_chains-1), dtype='uint8') * 51  # dark gray
    
    for ii in xrange(n_chains):
        # generate `plot_every` intermediate samples that we discard, because successive samples in the chain are too correlated
        test_idx = rng.randint(number_of_test_samples)
        
        samples = numpy.zeros((n_chains, n_input))

        visMean = test_set_x[test_idx,:]
        visSample = visMean
        for jj in xrange(n_samples):
            samples[jj,:] = visMean # show the mean, but use the sample for gibbs steps
            if jj == n_samples-1: break  # skip the last for speed
            plot_every = 2**jj  # exponentially increasing number of gibbs samples. max for n_samples=14 is 2^12
            for ss in xrange(plot_every):
                visMean, visSample = rbm.gibbs_vhv(visSample)[4:6]   # 4 for mean, 5 for sample

        print ' ... plotting sample ', ii
        image_data[:,(img_dim+1)*ii:(img_dim+1)*ii+img_dim] = tile_raster_images(
                X = imgPlotFunction(samples) if imgPlotFunction else samples,
                img_shape = (img_dim,img_dim),
                tile_shape = (n_samples, 1),
                tile_spacing = (1,1))
        if plotRawAlso:
            image_data_raw[:,(img_dim+1)*ii:(img_dim+1)*ii+img_dim] = tile_raster_images(
                    X = samples,
                    img_shape = (img_dim,img_dim),
                    tile_shape = (n_samples, 1),
                    tile_spacing = (1,1))

    image = Image.fromarray(image_data)
    image.save(os.path.join(output_dir, 'samples.png'))
    if plotRawAlso:
        image = Image.fromarray(image_data)
        image.save(os.path.join(output_dir, 'samplesRaw.png'))
    
    saveToFile(os.path.join(output_dir, 'rbm.pkl.gz'), rbm)
    
    return rbm, meanCosts



if __name__ == '__main__':
    resman.start('junk', diary = True)
    datasets = load_mnist_data('../data/mnist.pkl.gz', shared = False)
    print 'done loading.'
    test_rbm(datasets = datasets,
             training_epochs = 45,
             n_hidden = 500,
             learning_rate = .002,
             output_dir = resman.rundir,
             quickHack = False)
    resman.stop()
