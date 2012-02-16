#! /usr/bin/env python

import os, sys

import logging
#logging.basicConfig(filename='example.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y.%m.%d.%I.%M.%S %P')
#logger = logging.getLogger('simple_example')



class AutoLogger(object):
    '''A logging utility to override sys.stdout'''
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.log = logging.getLogger('autologger')
        self.log.setLevel(logging.DEBUG)
        self.fileHandler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s.%(msecs)d %(message)s', datefmt='%y.%m.%d.%H.%M.%S')
        self.fileHandler.setFormatter(formatter)
        self.log.addHandler(self.fileHandler)

    def write(self, message):
        self.stdout.write('writing message "%s"' % repr(message))
        self.stdout.write(message)
        self.log.info(message)

    def flush(self):
        self.stdout.flush()
        self.fileHandler.flush()

logger = AutoLogger('filename.log')
sys.stdout = logger
sys.stderr = sys.stdout


def main():
    print 'hi world!'
    print >>sys.stderr, 'hi errr'
    print 'hi world!'
    print 'hi world!'
    print
    print 'hi world!'
    #logging.info('hi log world!')
    #log2.info('hi log2 info')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')
    #log2.debug('hi log2 debug')


if __name__ == '__main__':
    main()


