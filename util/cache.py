#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

import pdb
import hashlib
import marshal
import os
from datetime import datetime
import time
from numpy import *
import cPickle as pickle

from dataLoaders import loadFromPklGz, saveToFile
from misc import mkdir_p



globalCacheDir     = '/tmp/pycache'     # Directory to use for caching
globalVerboseCache = False              # Print info about hits or misses
globalDisableCache = False              # Disable all caching



def memoize(function):
    '''Decorator to memoize function'''

    if globalDisableCache:

        def wrapper(*args, **kwargs):
            if globalVerboseCache:
                print ' -> cache.py: cache disabled by globalDisableCache'
            return function(*args, **kwargs)

    else:

        def wrapper(*args, **kwargs):
            argsHash = hashlib.sha1()

            # Hash the function name and function code itself (a bit overconservative)
            #print '  updating with function', function, 'hash', str(hash(function))
            #argsHash.update(str(hash(function)))
            #argsHash.update(function)
            argsHash.update(function.func_name)
            argsHash.update(marshal.dumps(function.func_code))

            # Hash *args
            for arg in args:
                try:
                    argsHash.update(str(hash(arg)))
                    #print '  updating with arg hash', str(hash(arg))
                except TypeError:
                    if type(arg) is ndarray:
                        argsHash.update(arg)
                        #print '  updating with ndarray', arg
                        #argsHash.update(pickle.dumps(arg, -1))  # another option
                    else:
                        raise

            # Hash **kwargs
            for key,value in sorted(kwargs.items()):
                argsHash.update(key)
                #print '  updating with kwarg key', key
                try:
                    argsHash.update(str(hash(value)))
                    #print '  updating with kwarg value hash', str(hash(value))
                except TypeError:
                    if type(value) is ndarray:
                        argsHash.update(value)
                        #print '  updating with kwarg ndarray', value
                        #argsHash.update(pickle.dumps(value, -1))  # another option
                    else:
                        raise

            digest = argsHash.hexdigest()

            cacheFilename    = '%s.%s.pkl.gz' % (digest[:16], function.func_name)
            # get a unique filename that does not affect any random number generators
            cacheTmpFilename = '.%s-%06d.tmp' % (cacheFilename, datetime.now().microsecond)
            cachePath    = os.path.join(globalCacheDir, cacheFilename[:2], cacheFilename)
            cacheTmpPath = os.path.join(globalCacheDir, cacheFilename[:2], cacheTmpFilename)

            try:
                start = time.time()
                (stats,result) = loadFromPklGz(cachePath)
                elapsedWall = time.time() - start
                if globalVerboseCache:
                    print ' -> cache.py: cache hit (%.04fs to load, saved %.04fs)' % (elapsedWall, stats['timeWall'] - elapsedWall)
            except IOError:
                startWall = time.time()
                startCPU  = time.clock()
                result = function(*args, **kwargs)
                elapsedWall = time.time() - startWall
                elapsedCPU  = time.clock() - startCPU

                if globalVerboseCache:
                    print ' -> cache.py: cache miss (%.04fs to compute)' % elapsedWall
                stats = {'func_name': function.func_name,
                         'timeWall': elapsedWall,
                         'timeCPU': elapsedCPU,
                         'saveDate': datetime.now(),
                         }

                mkdir_p(os.path.dirname(cachePath))
                saveToFile(cacheTmpPath, (stats,result), quiet=not globalVerboseCache)
                os.rename(cacheTmpPath, cachePath)

            return result

    return wrapper



def cached(function, *args, **kwargs):
    '''Return cached answer or compute and cache.'''

    memoizedFunction = memoize(function)
    return memoizedFunction(*args, **kwargs)



def invNoncached(mat, times = 1.0):
    return times * linalg.inv(mat)



@memoize
def invCached(mat, times = 1.0):
    return times * linalg.inv(mat)



def main():
    random.seed(0)
    a = random.rand(500,500)
    #print 'a is\n', a
    
    #ainv = invNoncached(a)
    #ainv = invCached(a)
    #ainv = linalg.inv(a)

    print 'computing ainv * 1.0'
    invCached(a, times = 1.0)
    print 'computing ainv * 2.0'
    invCached(a, 2.0)
    print 'computing cached(linalg.inv, a)'
    cached(linalg.inv, a)



if __name__ == '__main__':
    main()
