#! /usr/bin/env python

from numpy import *

from util.dataLoaders import saveToFile

'''
Creates random data!
'''


def main():
    for Nsamples in (50, 500, 5000, 50000):
        for Nw in (2, 3, 4, 6, 10, 15, 20, 25, 28):
            for color in (True, False):
                for seed,string in ((0, 'train'), (123456, 'test')):
                    random.seed(seed)
                    nColors = (3 if color else 1)
                    size = (Nw * Nw * nColors, Nsamples)
                    xx = random.uniform(0, 1, size)
                    saveToFile('../data/random/randomu01_%s_%d_%d_%dc.pkl.gz' % (string, Nw, Nsamples, nColors), xx)



if __name__ == '__main__':
    main()
