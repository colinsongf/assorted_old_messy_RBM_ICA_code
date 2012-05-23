#! /usr/bin/env ipythonwx

from numpy import *
from matplotlib import pyplot



def main():
    N = 200
    tt = linspace(0, 10, N)

    # make sources
    s1 = 4 + cos(tt)
    s2 = tt % 2

    pyplot.figure(1)
    pyplot.subplot(4,1,1)
    pyplot.title('original sources')
    pyplot.plot(tt, s1, 'bo-')
    pyplot.subplot(4,1,2)
    pyplot.plot(tt, s2, 'bo-')

    A = array([[3, 1], [-2, .3]])

    S = vstack((s1, s2)).T
    print 'S', S
    
    X = dot(S, A)
    print 'X', X

    x1 = X[:,0]
    x2 = X[:,1]

    pyplot.subplot(4,1,3)
    pyplot.title('observed signal')
    pyplot.plot(tt, x1, 'ro-')
    pyplot.subplot(4,1,4)
    pyplot.plot(tt, x2, 'ro-')

    pyplot.figure(2)
    pyplot.subplot(4,1,1)
    pyplot.title('original sources')
    pyplot.hist(s1)
    pyplot.subplot(4,1,2)
    pyplot.hist(s2)
    pyplot.subplot(4,1,3)
    pyplot.title('observed signal')
    pyplot.hist(x1)
    pyplot.subplot(4,1,4)
    pyplot.hist(x2)


    #pyplot.show()

    import ipdb; ipdb.set_trace()
    


if __name__ == '__main__':
    main()
    #pyplot.show()
    print 'shown'
    
    #import IPython; IPython.embed()
    #import IPython.core
    #IPython.core.ipapi.launch_new_instance(locals())
    #IPython.core.interactiveshell.InteractiveShell()

    #from IPython.core.debugger import Tracer
    #debug = Tracer()
    #debug()


    
