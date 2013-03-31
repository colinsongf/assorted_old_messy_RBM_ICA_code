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

stopwatch = Stopwatch()

def pt(st):
    '''Prepend the time since the start of the run to the given string'''

    return '%05.3f_%s' % (stopwatch.elapsed(), st)



class Counter(object):
    def __init__(self):
        self._count = -1

    def count(self):
        self._count += 1
        return self._count

counter = Counter()

def pc(st):
    '''Prepend a counter since the start of the run to the given string'''

    return '%03d_%s' % (counter.count(), st)
