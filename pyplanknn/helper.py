# this is basically a bunch of replacements for things broken in pypy
import numpy as np


def fromfile(filename, dtype=np.float16):
    # helper function to allow us to run under pypy
    try:
        return np.fromfile(filename, dtype=dtype)
    except NotImplementedError:
        # for numpypy
        return np.fromstring(open(filename, 'rb').read(), dtype=dtype)


def tofile(filename, arr):
    # helper function to allow us to run under pypy
    try:
        arr.tofile(filename)
    except AttributeError:
        # for numpypy
        open(filename, 'wb').write(arr.tostring())


def outer(i, j):
    #try:
    #    return np.outer(i, j)
    #except NotImplementedError:
    return np.multiply(i.ravel()[:, np.newaxis], j.ravel()[np.newaxis, :])
