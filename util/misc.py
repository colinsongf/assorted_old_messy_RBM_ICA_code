#! /usr/bin/env python

import os, errno
import time



def mkdir_p(path):
    '''Behaves like `mkdir -P` on Linux.
    From: http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python'''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



class Stopwatch(object):
    def __init__(self):
        self._start = time.time()

    def elapsed(self):
        return time.time() - self._start

globalStopwatch = Stopwatch()

def makePt(st, stopwatch = None):
    '''Prepend the time since the start of the run to the given string'''

    if string is None:
        stopwatch = globalStopwatch
    
    return '%05.3f_%s' % (stopwatch.elapsed(), st)

pt = lambda st : makePt(st)



class Counter(object):
    def __init__(self):
        self._count = -1

    def count(self):
        self._count += 1
        return self._count

globalCounter = Counter()

def makePc(st, counter = None):
    '''Prepend a counter since the start of the run to the given string'''

    if counter is None:
        counter = globalCounter
    
    return '%03d_%s' % (counter.count(), st)

class MakePc(object):
    def __init__(self, counter = None):
        if counter is None:
            counter = globalCounter
        self.counter = counter

    def __call__(self, st):
        return '%03d_%s' % (self.counter.count(), st)
    
pc = lambda st : makePc(st)
