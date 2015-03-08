from __future__ import print_function
import os
import random
import numpy as np
from pyplanknn.helper import fromfile
from pyplanknn.preprocess import read_train
from pyplanknn.network import Network, load

TRAIN_N = 1

print('Loading Network')
dirname = 'network_{:0=3d}'.format(TRAIN_N)
if os.path.exists(dirname):
    network = load(dirname)
else:
    network = Network()

print('Loading Data')
# smaller files sampled from full data to allow faster startup
x1 = fromfile('network_001/featdata.npy', dtype=np.uint8).reshape((13, 250, 250))
xo = fromfile('network_001/otherdata.npy', dtype=np.uint8).reshape((300, 250, 250))
#x, y = read_train()
#x1 = x[y == TRAIN_N]
#xo = x[y != TRAIN_N]
l = len(x1)  # number of cases to use

y = np.zeros((l, 2))
y[:, 0] = 1

print('Starting Training')

for i in range(10):
    # make xn the same length as x1 and add it on the end
    x_train = x1[random.sample(range(len(x1)), l)]
    x_train = x_train[:, \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1))), \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1)))]
    p = network(x_train.reshape(l, 62500) / 255 - 0.5, y, 0.1)

    print('Trained', i, np.sum(p, axis=0) / l)
    x_train = xo[random.sample(range(len(xo)), l)]
    x_train = x_train[:, \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1))), \
                      random.choice((slice(None, None, 1), \
                                     slice(None, None, -1)))]
    p = network(x_train.reshape(l, 62500) / 255 - 0.5, 1 - y, 0.01)
    print('Untrained', i, np.sum(p, axis=0) / l)

print('Saving')
network.save('network_001')


def plot_weights():
    import matplotlib.pyplot as plt
    WGT_NUM = 0

    a = fromfile('weights_00_62501.npy', dtype=np.float16)
    plt.imshow(a.reshape((62501, 400))[1:, WGT_NUM].reshape((250, 250)))
    plt.show()
