#! /usr/bin/env python
#! /usr/local/bin/ipython --gui=wx

import os, pdb, gzip, sys
import subprocess
import re
import pp
from numpy import *



def getXmlFiles(dataPath):
    ret = []
    pattern = re.compile('^[^_]+_[0-9]{5}\.xml$')
    for root, dirs, files in os.walk(dataPath):
        for fil in files:
            if pattern.search(fil):
                filename = os.path.join(root, fil)
                #print filename,
                #print 'yes'
                ret.append(filename)
                if len(ret) % 1000 == 0:
                    print 'xmlFiles so far:', len(ret)
            else:
                pass
                #print 'no'
    numXmlFiles = len(ret)

    return ret



def hyperneatExport(hnExec, shapesDat, xmlFile, outputPrefix):
    args = (hnExec, '-I', shapesDat, xmlFile, '-O', outputPrefix)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = proc.communicate()
    code = proc.wait()

    if code != 42:
        print out
        print err
        raise Exception('Expected exit code 42, but got %d instead. Args were: %s' % (code, repr(args)))



def main():
    if len(sys.argv) <= 3:
        print 'Usage:\n    # process xml files into txt shape files.\n    %s path_to_HN_exec path_to_shapes.dat path_to_directory_of_shapes' % (sys.argv[0])
        sys.exit(1)

    hnPath = sys.argv[1]
    shapesDatPath = sys.argv[2]
    dataPath = sys.argv[3]

    xmlFiles = getXmlFiles(dataPath)
    nXmlFiles = len(xmlFiles)
    print 'Processing', nXmlFiles, 'xml files'
    
    job_server = pp.Server(ncpus=9)
    jobs = []
    
    for ii, xmlFile in enumerate(xmlFiles):
        outputPrefix = '%s_EXPORT' % xmlFile[:-4]
        JOBS = True
        if JOBS:
            jobs.append((ii, xmlFile, 
                         job_server.submit(hyperneatExport,
                                           (hnPath, shapesDatPath, xmlFile, outputPrefix),
                                           modules=('subprocess',),
                                           ))
                        )
            #print 'started', ii
        else:
            hyperneatExport(hnPath, shapesDatPath, xmlFile, outputPrefix)
            #print 'done with', xmlFile

    for ii, xmlFile, job in jobs:
        results = job()
        #print ii, xmlFile, results, 'done'
        if ii % 100 == 0:
            print 'Finished %d/%d jobs' % (ii, nXmlFiles)
        if ii % 10000 == 0:
            job_server.print_stats()

    print
    
    job_server.print_stats()



if __name__ == '__main__':
    main()
