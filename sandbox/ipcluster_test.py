#! /usr/bin/env python

import time
from datetime import datetime, timedelta
import random
import sys
import ipdb as pdb
from IPython.parallel import Client

from ipcluster_helper import sleeper



def main():
    client = Client(profile='ssh')

    print 'ids:', client.ids
    
    view = client.load_balanced_view()

    #pdb.set_trace()

    tic = time.time()
    results = view.map(sleeper, [1] * 40)
    #results = map(sleeper, [1] * 4)

    print 'results:', results.get()
    print 'elapsed:', time.time()-tic


main()
