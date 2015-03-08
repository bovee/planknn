from itertools import product
import numpy as np
from numpy.fft.fftpack import rfft, fft, ifft, irfft


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


class ConvPoolLayer:
    def __init__(self, n_convs=10):
        #TODO: set weights manually
        self.max_locs_x = None
        self.max_locs_y = None
        self._input = None
        self.weights = np.random.normal(0, 1, (n_convs, 11, 11))
        self.momentum = np.zeros_like(self.weights)

        # window corners
        win_corn = list(product([0, 60, 120, 180], [0, 60, 120, 180]))
        win_corn += list(product([30, 90, 150], [30, 90, 150]))
        self.windows = []
        for win in win_corn:
            self.windows.append((win[0], win[1], 60, 60))

    def __call__(self, ims):
        self._input = ims
        output = np.empty((ims.shape[0], self.weights.shape[0], \
                           len(self.windows)), dtype=np.float32)
        self.max_locs_x = np.empty((ims.shape[0], self.weights.shape[0], \
                                    len(self.windows)), dtype=np.int16)
        self.max_locs_y = np.empty((ims.shape[0], self.weights.shape[0], \
                                    len(self.windows)), dtype=np.int16)
        for i, (wx, wy, ww, wh) in enumerate(self.windows):
            self.max_locs_x[:, :, i] = wx
            self.max_locs_y[:, :, i] = wy

        # precalculate the fourier transform of the weights
        # padding to 270 because it's a hamming number, so
        # calculation will be faster
        fweights = rfftn(self.weights, (270, 270))
        # precalculate the fourier transform of the images
        fims = rfftn(ims, (270, 270))

        for im_num, fim in enumerate(fims):
            # convolutional layer using fft
            cim = irfftn(fim * fweights, (270, 270))[:, 15:255, 15:255]
            #convolved size should be 240 x 240

            # followed by a max-pooling layer
            for win_num, (wx, wy, ww, wh) in enumerate(self.windows):
                for wgt_num in range(self.weights.shape[0]):
                    wind = cim[wgt_num, wx:wx + ww, wy:wy + wh]
                    loc = np.unravel_index(wind.argmax(), (wh, ww))
                    self.max_locs_x[im_num, wgt_num, win_num] += loc[0]
                    self.max_locs_y[im_num, wgt_num, win_num] += loc[1]
                    output[im_num, wgt_num, win_num] = cim[wgt_num, \
                                                           loc[0] + ww, \
                                                           loc[1] + wh]
        return output.reshape((output.shape[0], \
                               output.shape[1] * output.shape[2]))

    def error_in(self, error_out):
        #TODO: should fix this, but its faster to not calculate it
        error_in = None
        #error_in = np.zeros_like(self._input)
        #for i in range(error_out.shape[0]):
        #    for loc, err in zip(self.max_locs[i], error_out[i]):
        #        error_in[i, loc[0]:loc[0] + 11, loc[1]:loc[1] + 11] += err

        return error_in

    def train(self, error_out, learning_rate):
        error_out = error_out.reshape((error_out.shape[0], \
                                       self.weights.shape[0], \
                                       len(self.windows)))
        min_err = np.abs(error_out).argmin(axis=2)

        # TODO: check again there's no easy way to use the results of an
        # argmin along an axis in a multidim array to get the results out
        ixs = np.mgrid[0:min_err.shape[0], \
                       0:min_err.shape[1]].reshape((2, np.prod(min_err.shape)))
        err_ixs = np.vstack([ixs, min_err.flatten()])
        err_ixs = np.ravel_multi_index(err_ixs, self.max_locs_x.shape)

        exs = self.max_locs_x.flatten()[err_ixs].reshape(min_err.shape)
        eys = self.max_locs_y.flatten()[err_ixs].reshape(min_err.shape)

        for wgt_num in range(error_out.shape[1]):
            gradients = np.empty((error_out.shape[0], 11, 11))
            for im_num in range(error_out.shape[0]):
                ex, ey = exs[im_num, wgt_num], eys[im_num, wgt_num]
                im_clip = self._input[im_num, ex:ex + 11, ey:ey + 11]
                #TODO: multiply by np.sign(min_err[im_num, wgt_num]) ???
                gradients[im_num, :, :] = im_clip / \
                        (0.5 + np.abs(min_err[im_num, wgt_num]))

            self.momentum[wgt_num] = 0.9 * self.momentum[wgt_num] - \
                    learning_rate * np.mean(gradients, axis=0)
        self.weights += self.momentum
