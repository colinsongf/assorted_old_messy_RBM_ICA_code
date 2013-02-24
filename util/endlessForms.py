#! /usr/bin/env python



def getEFFindUrl(baseUrl, runIdOrAll, genSerial = None, orgSerial = None):
    '''Get the URL to find a shape based on its runId, genSerial, and orgSerial.

    Returns URLs like
    http://devj.cornell.endlessforms.com/debug/find/hicrvb9ka3-1-5'''

    if genSerial is None:
        # all packed into runIdOrAll
        assert(len(runIdOrAll) == 3)
        runId, genSerial, orgSerial = runIdOrAll
    else:
        # args given separately
        runId = runIdOrAll

    return '%s/debug/find/%s-%d-%d' % (baseUrl, runId, genSerial, orgSerial)
