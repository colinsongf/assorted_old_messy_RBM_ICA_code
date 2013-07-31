#! /usr/bin/env ipythonpl

import sys
import os
import re
import ipdb as pdb
import argparse
from IPython import embed
#from numpy import *
#from PIL import Image
from matplotlib import pyplot

from GitResultsManager import resman

from util.misc import trimCommon
from util.plotting import looser



patterns = [
    r'Micro avg pr = (?P<mean>[-.0-9]+) stdev: (?P<stddev>[-.0-9]+)',
    r'Micro avg rc = (?P<mean>[-.0-9]+) stdev: (?P<stddev>[-.0-9]+)',
    r'Macro avg pr = (?P<mean>[-.0-9]+) stdev: (?P<stddev>[-.0-9]+)',
    r'Macro avg rc = (?P<mean>[-.0-9]+) stdev: (?P<stddev>[-.0-9]+)',
    r'Micro Averaged Precision Recall:\n1.00\tprec: (?P<precision>[-.0-9]+)\trecall: (?P<recall>[-.0-9]+)',
    r'Macro Averaged Precision Recall:\n1.00\tprec: (?P<precision>[-.0-9]+)\trecall: (?P<recall>[-.0-9]+)',
    ]



def visPerf(modelFiles, saveDir = None):
    fileData = parseFiles(modelFiles)
    shortNames = trimCommon([name for name,info in fileData])

    print 'HERE'
    print 'Next: save figure to file, do sub-activity as well as just affordance'

    pyplot.figure()
    #pyplot.hold(True)
    for ii in range(6):
        pyplot.subplot(3,2,ii+1)
        if 'mean' in fileData[0][1]['affordance'][ii]:
            for fileIdx in range(len(fileData)):
                mn     = float(fileData[fileIdx][1]['affordance'][ii]['mean'])
                stddev = float(fileData[fileIdx][1]['affordance'][ii]['stddev'])
                pyplot.errorbar([fileIdx], mn, yerr=stddev, fmt='o', color='b')
            axis(looser(axis()))
        elif 'precision' in fileData[0][1]['affordance'][ii]:
            minval = 9999
            for fileIdx in range(len(fileData)):
                prec = float(fileData[fileIdx][1]['affordance'][ii]['precision'])
                rec  = float(fileData[fileIdx][1]['affordance'][ii]['recall'])
                pyplot.bar(fileIdx, prec, color='r')
                pyplot.bar(fileIdx+len(fileData)+1, rec, color='g')
                if prec < minval: minval = prec
                if rec  < minval: minval = rec
            ax = list(axis())
            ax[2] = minval - .1 * (ax[3]-minval)
            axis(ax)
    #pyplot.subplot(3,1,1)
    pyplot.suptitle('\n'.join(shortNames))
    pdb.set_trace()



def parseFiles(modelFiles):
    ret = []
    for modelFile in modelFiles:
        with open(modelFile) as ff:
            text = ff.read()
        pos = 0

        fileInfo = {'affordance':[], 'sub-activity': []}

        # Walk through the file looking for one pattern at a time.
        for ii in range(12):
            patternIdx = ii % 6
            mm = re.search(patterns[patternIdx], text[pos:])
            if not mm:
                raise Exception('Parse error for file: %s' % modelFile)
            pos += mm.end()

            if ii < 6:
                fileInfo['affordance'].append(mm.groupdict())
            else:
                fileInfo['sub-activity'].append(mm.groupdict())

        ret.append((modelFile, fileInfo))

    #print 'Parsed:'
    #print repr(dat)
    #pdb.set_trace()
    
    return ret



def main():
    parser = argparse.ArgumentParser(description='Visualize performance of different human-activity-detection runs using the generated avg_pr.model.c0.1.e0.01.w3 files (or similar)')
    parser.add_argument('--name', type = str, default = 'junk',
                        help = 'Name for GitResultsManager results directory (default: junk)')
    parser.add_argument('--quick', action='store_true', help = 'Enable quick mode (default: off)')
    parser.add_argument('--nodiary', action='store_true', help = 'Disable diary (default: diary is on)')

    parser.add_argument('avg_pr_model_file', type = str, nargs = '+',
                        help = 'Which avg_pr.model.c0.1.e0.01.w3 (or similar) files to load. Must give at least one.')

    args = parser.parse_args()

    resman.start(args.name, diary = not args.nodiary)

    visPerf(saveDir = resman.rundir, modelFiles = args.avg_pr_model_file)
    
    resman.stop()



if __name__ == '__main__':
    main()
