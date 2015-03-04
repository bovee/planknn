"""
Stripped down FFT functions from numpy's fftpack.

TODO: find another C implementation of these that works with pypy.
"""
#from numpy.fft.fftpack import irfftn, rfftn
from numpy.fft.fftpack import rfft, fft, ifft, irfft
import numpy as np


def rfftn(a, s):
    a = a.astype(float)
    a = rfft(a, s[-1], -1)
    a = fft(a, s[-2], -2)
    return a


def irfftn(a, s):
    a = a.astype(complex)
    a = ifft(a, s[-2], -2)
    a = irfft(a, s[-1], -1)
    return a
