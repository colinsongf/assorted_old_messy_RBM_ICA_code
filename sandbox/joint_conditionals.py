#! /usr/bin/env ipythonpl


import ipdb as pdb
from pylab import *
from numpy import *


NN = 10
vals = range(NN)

def p_x1_x2(x2):
    ret = array([1.] * NN)
    ret[(x2+1)%NN] += 10
    return ret / ret.sum()
    
def p_x2_x1(x1):
    ret = array([1.] * NN)
    ret[(2*x1)%NN] += 10
    return ret / ret.sum()
    
def main():
    x1, x2 = [0, 0]
    states = []

    for ii in xrange(50000):
        states.append([x1, x2])
        if rand() > .5:
        #if ii % 2 == 0:
            x1 = random.choice(vals, p = p_x1_x2(x2))
        else:
            x2 = random.choice(vals, p = p_x2_x1(x1))

            stationary = zeros((NN,NN), dtype='float')
    for state in states:
        stationary[state[0], state[1]] += 1
    stationary /= stationary.sum()

    #print states
    print stationary

    gray()
    imshow(stationary, interpolation='nearest', origin='lower')
    title('stationary')
    savefig('stationary.png')
    #figure()
    #imshow(log(stationary), interpolation='nearest', origin='lower')
    #title('log(stationary)')
    #savefig('log_stationary.png')

    #pdb.set_trace()



if __name__ == '__main__':
    main()
