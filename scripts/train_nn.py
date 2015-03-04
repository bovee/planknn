#!/usr/bin/env python3
import os
import random
import numpy as np
from pyplanknn.preprocess import read_train
from pyplanknn.network import Network, load

WEIGHTS_FILE = '../../test_dump.txt'
N_TRAINS = 50
N_RANDS = 1


if __name__ == '__main__':
    if os.path.exists(WEIGHTS_FILE):
        model = load(WEIGHTS_FILE)
    else:
        model = Network(16, [512, 512, 512, 512], 121)

    x, x_class = read_train()
    y = np.zeros((x_class.shape[0], 121))
    for i, c in enumerate(x_class):
        y[i, c] = 1.

    training = np.empty((121, N_TRAINS), dtype=int)
    for i in range(121):
        training[i] = np.random.choice(np.nonzero(x_class == i)[0], \
                                    N_TRAINS, replace=True)

    for i in range(N_TRAINS):
        # adding in noisy samples is equivalent to an L2 penalty on our weights
        x_train = np.concatenate((x[training[:, i]] / 255, \
                                  np.random.random((N_RANDS, 250, 250))), axis=0)
        y_train = np.concatenate((y[training[:, i]], \
                                  np.ones((N_RANDS, 121)) / 121.), axis=0)

        # randomly flip images
        x_train = x_train[:, \
                        random.choice((slice(None, None, 1), \
                                        slice(None, None, -1))), \
                        random.choice((slice(None, None, 1), \
                                        slice(None, None, -1)))]
        model(x_train, y_train, 1)

    model.save(WEIGHTS_FILE)
