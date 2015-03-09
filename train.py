from __future__ import print_function
import random
import numpy as np
from pyplanknn.preprocess import read_train
from pyplanknn.network import Network, load
from pyplanknn.submit import multiclass_log_loss

TRAIN_N = 1
N_TRAINS = 60
N_RANDS = 1

print('Loading Data')
# smaller files sampled from full data to allow faster startup
x, x_class = read_train()
y = np.zeros((x_class.shape[0], 121))
for i, c in enumerate(x_class):
    y[i, c] = 1.

print('Loading Network')
dirname = 'network_{:0=3d}'.format(TRAIN_N)
network = load(dirname)
if network is None:
    print('Create New Network')
    network = Network()
    for i in range(121):
        network.layers[0].weights[1:, i] += \
            0.01 * (x[x_class == i].mean(axis=0).flatten() - 127.5) / 127.5

print('Starting Training')

training = np.empty((121, N_TRAINS), dtype=int)
for i in range(121):
    training[i] = np.random.choice(np.nonzero(x_class == i)[0], \
                                   N_TRAINS, replace=True)

for i in range(N_TRAINS):
    # adding in noisy samples is equivalent to an L2 penalty on our weights
    x_train = np.concatenate((x[training[:, i]], \
                              np.random.random((N_RANDS, 250, 250))), axis=0)
    y_train = np.concatenate((np.eye(121), \
                              np.ones((N_RANDS, 121)) / 121.), axis=0)
    #y_train = np.concatenate((y[training[:, i]], \
    #                          np.ones((N_RANDS, 121)) / 121.), axis=0)

    # randomly flip images
    x_train = x_train[:, \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1))), \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1)))]

    x_train = (x_train - 127.5).reshape(-1, 62500)
    x_train /= 127.5

    res, _ = network(x_train, y_train, 0.01)

    print('Trained', i, multiclass_log_loss(np.arange(121), res[:121]))

print('Saving')
network.save(dirname)


def plot_weights(network):
    import matplotlib.pyplot as plt
    w = network.layers[0].weights.reshape((62501, 400))[1:]
    sw = np.sum(w[:, :121], axis=1)
    plt.subplot(2, 1, 1)
    plt.imshow(sw.reshape((250, 250)))
    plt.subplot(2, 1, 2)
    sw = np.sum(w[:, 121:], axis=1)
    plt.imshow(sw.reshape((250, 250)))
    #WGT_NUM = 0
    #w = network.layers[0].weights
    #plt.imshow(w.reshape((62501, 400))[1:, WGT_NUM].reshape((250, 250)))
    plt.show()
