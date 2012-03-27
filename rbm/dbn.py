#! /usr/bin/env python

# [JBY] Modified from http://deeplearning.net/tutorial/DBN.html

import os
import numpy, time, cPickle, gzip, os, sys
import pdb

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM



class DBN(object):
    '''Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the 
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    '''

    def __init__(self, numpyRng, theanoRng = None, nInputs = 784,
                 hiddenLayerSizes = [500,500], nOutputs = 10):
        '''This class is made to support a variable number of layers. 

        :type numpyRng: numpy.random.RandomState
        :param numpyRng: numpy random number generator used to draw initial 
                    weights

        :type theanoRng: theano.tensor.shared_randomstreams.RandomStreams
        :param theanoRng: Theano random generator; if None is given one is 
                           generated based on a seed drawn from `rng`

        :type nInputs: int
        :param nInputs: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain 
                               at least one value

        :type nOutputs: int
        :param nOutputs: dimension of the output of the network
        '''

        self.nLayers = length(hiddenLayerSizes)

        if not theanoRng:
            theanoRng = RandomStreams(numpyRng.randint(2**30))
        self.numpyRng  = numpyRng
        self.theanoRng = theanoRng

        layerSizes = [nInputs] + hiddenLayerSizes
        self.rbms = []
        for ii in range(len(layerSizes) - 1):
            self.rbms.append(RBM(layerSizes[ii], layerSizes[ii+1], self.numpyRng, self.theanoRng))
        print 'RBMs are:', self.rbms


    def pretraining_functions(self, train_set_x, batch_size,k):
        ''' Generates a list of functions, for performing one step of gradient descent at a
        given layer. The function will require as input the minibatch index, and to train an
        RBM you just need to iterate, calling the corresponding function on all minibatch
        indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        '''

        # index to a [mini]batch
        index            = T.lscalar('index')   # index to a minibatch
        learning_rate    = T.scalar('lr')    # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost,updates = rbm.get_cost_updates(learning_rate, persistent=None, k =k)

            # compile the theano function    
            fn = theano.function(inputs = [index, 
                              theano.Param(learning_rate, default = 0.1)],
                    outputs = cost, 
                    updates = updates,
                    givens  = {self.x :train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
 

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of finetuning, a function
        `validate` that computes the error on a batch from the validation set, and a function
        `test` that computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;  the has to contain three
        pairs, `train`, `valid`, `test` in this order, where each pair is formed of two Theano
        variables, one for the datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size

        index   = T.lscalar('index')    # index to a [mini]batch 

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate

        train_fn = theano.function(inputs = [index], 
              outputs =   self.finetune_cost, 
              updates = updates,
              givens  = {
                self.x : train_set_x[index*batch_size:(index+1)*batch_size],
                self.y : train_set_y[index*batch_size:(index+1)*batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens = {
                   self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                   self.y: test_set_y[index*batch_size:(index+1)*batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens = {
                 self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                 self.y: valid_set_y[index*batch_size:(index+1)*batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score



    # Old def get_cost_updates(self, train_x, lr = 0.1, persistent=None, k =1):
    ############# FROM RBM #####################
    def train(self, train_x, lr = 0.1, persistent = None, k = 1, metrics = False, plotWeights = False):
        """
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
        """

        pdb.set_trace() # HERE

        
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

    def __str__(self):
        return 'DBN with RBMs ' + repr(self.rbms)


def test_DBN( finetune_lr = 0.1, pretraining_epochs = 100, \
              pretrain_lr = 0.01, k = 1, training_epochs = 1000, \
              dataset='../data/mnist.pkl.gz', batch_size = 10):
    '''
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage 
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer 
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    '''


    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    print 'Types are:'
    print 'train_set_x', train_set_x.type
    print 'train_set_y', train_set_y.type
    pdb.set_trace()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpyRng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpyRng = numpyRng, nInputs = 28*28, 
              hiddenLayerSizes = [1000,1000,1000],
              nOutputs = 10)
    

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(
            train_set_x   = train_set_x, 
            batch_size    = batch_size,
            k             = k) 

    print '... pre-training the model'
    start_time = time.clock()  
    ## Pre-train layer-wise 
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index = batch_index, 
                         lr = pretrain_lr ) )
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),numpy.mean(c)
 
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions ( 
                datasets = datasets, batch_size = batch_size, 
                learning_rate = finetune_lr) 

    print '... finetunning the model'
    # early-stopping parameters
    patience              = 4*n_train_batches # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = numpy.inf
    test_score           = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_fn(minibatch_index)
        iter    = epoch * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0: 
            
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_train_batches, \
                    this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_train_batches,
                              test_score*100.))


        if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))



if __name__ == '__main__':
    test_DBN()
