#! /usr/bin/env python

import time
from datetime import datetime, timedelta
import random
import sys
import ipdb as pdb
from IPython.parallel import Client



def sleeper(sec = 1):
    import time
    
    print 'at', time.time()
    time.sleep(sec)
    return sec
