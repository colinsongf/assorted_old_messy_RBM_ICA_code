#! /usr/bin/env ipythonpl

from matplotlib import *
from numpy import *
import Image
from IPython import embed

from stackedLayers import *
from layers import *
from util.plotting import *

from sandbox import *



def main():
    notes = '''Notes:
    raw = dl.getData((10,10), 10000, 0)
    cent = (raw.T - raw.mean(1)).T
    dat = cent / sqrt(sum(cent**2, 0) + (1e-12))
    '''
    print notes
    
    embed()



if __name__ == '__main__':
    main()
