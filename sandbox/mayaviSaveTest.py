#! /usr/local/bin/ipython --gui=wx

# Run like:
#  DISPLAY=:0 time ./mayaviSaveTest.py foo

import sys
from mayavi import mlab

#mlab.options.offscreen = True
#raw_input('about to plot')
for ii in range(5):
    mlab.clf()
    mlab.test_contour3d()

    #raw_input('about to save')
    mlab.savefig('test_%s_%02d.png' % (sys.argv[1], ii))
    #mlab.savefig('test_%s.jpg' % sys.argv[1])
    #mlab.savefig('test_%s.ps' % sys.argv[1])
    #mlab.savefig('test_%s.pdf' % sys.argv[1])
