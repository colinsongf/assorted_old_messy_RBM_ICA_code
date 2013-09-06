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
    resman.start('junk', diary = False)
    saveDir = resman.rundir

    dirs = [name for name in os.listdir('results') if os.path.isdir(os.path.join('results', name))]
    print 'last few results:'
    for dir in sorted(dirs)[-10:]:
        print '  ' + dir
    
    notes = '''Notes:
    raw = dl.getData((10,10), 10000, 0)
    cent = (raw.T - raw.mean(1)).T
    dat = cent / sqrt(sum(cent**2, 0) + (1e-12))
    '''
    print notes
    
    embed()

    resman.stop()



if __name__ == '__main__':
    main()
