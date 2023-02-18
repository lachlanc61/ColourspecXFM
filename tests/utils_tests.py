import pytest
import sys, os
import fnmatch

def findin(fname, datafiles):
    """
    searches a LocalPath for a file ending with fname, and returns LocalPath for that file

    note: fname should include extension

    raises error if multiple matches in LocalPath
    """

    flist = datafiles.listdir()

    nmatched = 0

    matched = None

    for i in flist:
        if fnmatch.fnmatch(i, "*"+fname):
            matched = i
            nmatched += 1

    if nmatched > 1:
        raise ValueError('Multiple matches in datafiles.listdir()')
    elif nmatched <= 0:
        raise ValueError(f'No matches found for {fname}')

    return matched