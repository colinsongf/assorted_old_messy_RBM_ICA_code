#! /usr/bin/env python

import os, sys, time, logging, subprocess, datetime, stat
import gzip
import cPickle as pickle

import numpy
from numpy import log, exp
import matplotlib.ticker as MT
import matplotlib.cm as CM



class DuckStruct(object):
    '''Use to store anything!'''
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        rep = ['%s=%s' % (k, repr(v)) for k,v in self.__dict__.items()]
        return 'DuckStruct(%s)' % ', '.join(rep)



def saveToFile(filename, obj, quiet = False):
    ff = gzip.open(filename, 'wb')
    pickle.dump(obj, ff, protocol = -1)
    if not quiet:
        print 'saved to', filename
    ff.close()



def loadFromFile(filename):
    with gzip.open(filename, 'rb') as ff:
        ret = pickle.load(ff)
    return ret



def load_mnist_data(dataset, shared = True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    if not shared:
        return train_set, valid_set, test_set

    def shared_dataset(data_xy):
        ''' Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        '''
        import theano
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval



def loadUpsonData(trainFile, testFile):
    ''' Loads the dataset and returns in the expected train,valid,test format.'''

    # Load the dataset
    ff = gzip.open(trainFile,'rb')
    train_set = pickle.load(ff)
    ff.close()
    ff = gzip.open(testFile,'rb')
    test_set = pickle.load(ff)
    ff.close()

    # no validation set, no y (purely unsupervised)
    return [train_set, None], [numpy.array([]), None], [test_set, None]



#def fmtSeconds(sec):
#    sign = ''
#    if sec < 0:
#        sign = '-'
#        sec = -sec
#    hours, remainder = divmod(sec, 3600)
#    minutes, seconds = divmod(remainder, 60)
#    if hours > 0:
#        return sign + '%d:%02d:%02d' % (hours, minutes, int(seconds)) + ('%.3f' % (seconds-int(seconds)))[1:]
#    elif minutes > 0:
#        return sign + '%d:%02d' % (minutes, int(seconds)) + ('%.3f' % (seconds-int(seconds)))[1:]
#    else:
#        return sign + '%d' % int(seconds) + ('%.3f' % (seconds-int(seconds)))[1:]
#
#
#
#class OutstreamHandler(object):
#    def __init__(self, writeHandler, flushHandler):
#        self.writeHandler = writeHandler
#        self.flushHandler = flushHandler
#
#    def write(self, message):
#        self.writeHandler(message)
#
#    def flush(self):
#        self.flushHandler()
#
#
#
#class OutputLogger(object):
#    '''A logging utility to override sys.stdout'''
#
#    '''Buffer states'''
#    class BState:
#        EMPTY  = 0
#        STDOUT = 1
#        STDERR = 2
#            
#    def __init__(self, filename):
#        self.stdout = sys.stdout
#        self.stderr = sys.stderr
#        self.log = logging.getLogger('autologger')
#        self.log.propagate = False
#        self.log.setLevel(logging.DEBUG)
#        self.fileHandler = logging.FileHandler(filename)
#        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', datefmt='%y.%m.%d.%H.%M.%S')
#        self.fileHandler.setFormatter(formatter)
#        self.log.addHandler(self.fileHandler)
#
#        self.stdOutHandler = OutstreamHandler(self.handleWriteOut,
#                                              self.handleFlushOut)
#        self.stdErrHandler = OutstreamHandler(self.handleWriteErr,
#                                              self.handleFlushErr)
#        self.buffer = ''
#        self.bufferState = self.BState.EMPTY
#        self.started = False
#
#
#    def startCapture(self):
#        if self.started:
#            raise Exception('ERROR: OutputLogger capture was already started.')
#        self.started = True
#        sys.stdout = self.stdOutHandler
#        sys.stderr = self.stdErrHandler
#
#    def finishCapture(self):
#        if not self.started:
#            raise Exception('ERROR: OutputLogger capture was not started.')
#        self.started = False
#        self.flush()
#        sys.stdout = self.stdout
#        sys.stderr = self.stderr
#
#    def handleWriteOut(self, message):
#        self.write(message, self.BState.STDOUT)
#        
#    def handleWriteErr(self, message):
#        self.write(message, self.BState.STDERR)
#
#    def handleFlushOut(self):
#        self.flush()
#        
#    def handleFlushErr(self):
#        self.flush()
#        
#    def write(self, message, destination):
#        if destination == self.BState.STDOUT:
#            self.stdout.write(message)
#        else:
#            self.stderr.write(message)
#        
#        if destination == self.bufferState or self.bufferState == self.BState.EMPTY:
#            self.buffer += message
#            self.bufferState = destination
#        else:
#            # flush and change buffer
#            self.flush()
#            assert(self.buffer == '')
#            self.bufferState = destination
#            self.buffer = '' + message
#        if '\n' in self.buffer:
#            self.flush()
#
#    def flush(self):
#        self.stdout.flush()
#        self.stderr.flush()
#        if self.bufferState != self.BState.EMPTY:
#            if len(self.buffer) > 0 and self.buffer[-1] == '\n':
#                self.buffer = self.buffer[:-1]
#            if self.bufferState == self.BState.STDOUT:
#                for line in self.buffer.split('\n'):
#                    self.log.info('  ' + line)
#            elif self.bufferState == self.BState.STDERR:
#                for line in self.buffer.split('\n'):
#                    self.log.info('* ' + line)
#            self.buffer = ''
#            self.bufferState = self.BState.EMPTY
#        self.fileHandler.flush()
#
#
#
#def gitExecutable():
#    return 'git'
#
#
#
#def runCmd(args):
#    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    out,err = proc.communicate()
#    code = proc.wait()
#
#    if code != 0:
#        print out
#        print err
#        raise Exception('Got error from running command with args ' + repr(args))
#
#    return out, err
#
#
#
#def gitLastCommit():
#    return runCmd(('git', 'rev-parse', '--short', 'HEAD'))[0].strip()
#
#
#
#def gitCurrentBranch():
#    out, err = runCmd(('git', 'branch'))
#    for line in out.split('\n'):
#        if len(line) > 2 and line[0] == '*':
#            return line[2:]
#    raise Exception('Error getting current branch from git stdout/stderr %s, %s.' % (repr(out), repr(err)))
#
#
#
#def gitStatus():
#    return runCmd(('git', 'status'))[0].strip()
#
#
#
#def gitDiff(color = False):
#    if color:
#        return runCmd(('git', 'diff', '--color'))[0].strip()
#    else:
#        return runCmd(('git', 'diff'))[0].strip()
#
#
#
#def hostname():
#    return runCmd('hostname')[0].strip()
#
#
#
#RESULTS_SUBDIR = 'results'
#
#class ResultsManager(object):
#    '''Creates directory for results'''
#
#    def __init__(self, resultsSubdir = None):
#        self._resultsSubdir = resultsSubdir
#        if self._resultsSubdir is None:
#            self._resultsSubdir = RESULTS_SUBDIR
#        if not stat.S_ISDIR(os.stat(self._resultsSubdir).st_mode):
#            raise Exception('Please create the results directory "%s" first.' % resultsSubdir)
#        self._name = None
#        self._outLogger = None
#        self.diary = None
#        
#    def start(self, description = '', diary = True):
#        if self._name is not None:
#            self.finish()
#        self.diary = diary
#        lastCommit = gitLastCommit()
#        curBranch = gitCurrentBranch()
#        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
#
#        basename = '%s_%s_%s' % (timestamp, lastCommit, curBranch)
#        if description:
#            basename += '_%s' % description
#        success = False
#        ii = 0
#        while not success:
#            name = basename + ('_%d' % ii if ii > 0 else '')
#            try:
#                os.mkdir(os.path.join(self._resultsSubdir, name))
#                success = True
#            except OSError:
#                print >>sys.stderr, name, 'already exists, appending suffix to name'
#                ii += 1
#        self._name = name
#
#        if self.diary:
#            self._outLogger = OutputLogger(os.path.join(self.rundir, 'diary'))
#            self._outLogger.startCapture()
#
#        self.startWall = time.time()
#        self.startProc = time.clock()
#
#        # print the command that was executed
#        print '  Logging directory:', self.rundir
#        print '        Command run:', ' '.join(sys.argv)
#        print '           Hostname:', hostname()
#        print '  Working directory:', os.getcwd()
#        if not self.diary:
#            print '<diary not saved>'
#            # just log these three lines
#            with open(os.path.join(self.rundir, 'diary'), 'w') as ff:
#                print >>ff, '  Logging directory:', self.rundir
#                print >>ff, '        Command run:', ' '.join(sys.argv)
#                print >>ff, '           Hostname:', hostname()
#                print >>ff, '  Working directory:', os.getcwd()
#                print >>ff, '<diary not saved>'
#
#        with open(os.path.join(self.rundir, 'gitinfo'), 'w') as ff:
#            ff.write('%s %s\n' % (lastCommit, curBranch))
#        with open(os.path.join(self.rundir, 'gitstat'), 'w') as ff:
#            ff.write(gitStatus() + '\n')
#        with open(os.path.join(self.rundir, 'gitdiff'), 'w') as ff:
#            ff.write(gitDiff() + '\n')
#        with open(os.path.join(self.rundir, 'gitcolordiff'), 'w') as ff:
#            ff.write(gitDiff(color=True) + '\n')
#
#    def stop(self):
#        # TODO: output timing info?
#        if not self.diary:
#            # just log these couple lines before resetting our name
#            with open(os.path.join(self.rundir, 'diary'), 'a') as ff:
#                print >>ff, '       Wall time: ', fmtSeconds(time.time() - self.startWall)
#                print >>ff, '  Processor time: ', fmtSeconds(time.clock() - self.startProc)
#        self._name = None
#        print '       Wall time: ', fmtSeconds(time.time() - self.startWall)
#        print '  Processor time: ', fmtSeconds(time.clock() - self.startProc)
#        if self.diary:
#            self._outLogger.finishCapture()
#            self._outLogger = None
#
#
#    @property
#    def rundir(self):
#        if self._name:
#            return os.path.join(self._resultsSubdir, self._name)
#
#    @property
#    def runname(self):
#        return self._name
#


def imagesc(W, pixwidth=1, ax=None, grayscale=True):
    '''
    Do intensity plot, similar to MATLAB imagesc()

    W = intensity matrix to visualize
    pixwidth = size of each W element
    ax = matplotlib Axes to draw on 
    grayscale = use grayscale color map

    Rely on caller to .show()

    Simple matrix intensity plot, similar to MATLAB imagesc()
    
    David Andrzejewski (david.andrzej@gmail.com)
    From: https://gist.github.com/940072
    '''

    # import at last minute to allow user to change settings via previous matplotlib.use()
    from matplotlib import pyplot

    # N = rows, M = column
    (N, M) = W.shape 
    # Need to create a new Axes?
    if(ax == None):
        ax = pyplot.figure().gca()
    # extents = Left Right Bottom Top
    exts = (0, pixwidth * M, 0, pixwidth * N)
    if(grayscale):
        ax.imshow(W,
                  interpolation='nearest',
                  cmap=CM.gray,
                  extent=exts)
    else:
        ax.imshow(W,
                  interpolation='nearest',
                  extent=exts)

    ax.xaxis.set_major_locator(MT.NullLocator())
    ax.yaxis.set_major_locator(MT.NullLocator())
    return ax



def imagescDemo():
    # import at last minute to allow user to change settings via previous matplotlib.use()
    from matplotlib import pyplot

    # Define a synthetic test dataset
    testweights = numpy.array([[0.25, 0.50, 0.25, 0.00],
                            [0.00, 0.50, 0.00, 0.00],
                            [0.00, 0.10, 0.10, 0.00],
                            [0.00, 0.00, 0.25, 0.75]]) - 10 * 3
    # Display it
    ax = scaledimage(testweights)
    pyplot.show()



def looser(ax, frac = .05, semilogx = False, semilogy = False, loglog = False):
    '''Returns a loosened version of the axies specified in ax'''

    ax = list(ax)

    if loglog:
        semilogx = True
        semilogy = True

    if semilogx:
        ax[0] = log(ax[0])
        ax[1] = log(ax[1])
    if semilogy:
        ax[2] = log(ax[2])
        ax[3] = log(ax[3])
        
    ax = [(1+frac) * ax[0] - frac * ax[1], (1+frac) * ax[1] - frac * ax[0],
          (1+frac) * ax[2] - frac * ax[3], (1+frac) * ax[3] - frac * ax[2]]

    if semilogx:
        ax[0] = exp(ax[0])
        ax[1] = exp(ax[1])
    if semilogy:
        ax[2] = exp(ax[2])
        ax[3] = exp(ax[3])
    
    return tuple(ax)



if __name__ == '__main__':
    import time
    from ResultsManager import resman
    
    logger = OutputLogger('filename.log')
    print 'not logged'
    logger.startCapture()
    for ii in range(3):
        print '  this is logged...'
        print >>sys.stderr, '  oh, stderr too'
        time.sleep(1)
    print 'finished, turning logging off.'
    logger.finishCapture()
    print 'this is not logged'

    resman.start()
    print 'this is being logged to the %s directory' % resman.rundir
    time.sleep(1)
    print 'this is being logged to the %s directory' % resman.rundir
    time.sleep(1)
    print 'this is being logged to the %s directory' % resman.rundir
    resman.stop()
