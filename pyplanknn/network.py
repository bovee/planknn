import os
import numpy as np
from pyplanknn.layer import Layer
from pyplanknn.helper import fromfile, tofile


class Network:
    """
    A neural network comprised of several layers interconnected.
    """
    def __init__(self, n_in=62500, hidden=2, n_outs=121):
        """
        Create a series of layers.
        """
        if n_in is None:
            # allow us to create an empty network we can fill
            # with layers ourselves
            self.layers = []
            return

        if isinstance(hidden, int):
            args = hidden * [400] + [n_outs]
        else:
            args = hidden + [n_outs]

        args = iter(args)
        prev = next(args)
        self.layers = [Layer(n_in, prev, 'tanh')]
        for i in args:
            self.layers.append(Layer(prev, i, 'tanh'))
            prev = i

        # add a softmax layer at the end
        self.layers.append(Layer(i, i, 'softmax'))

    def __call__(self, vector, expected=None, learning_rate=0.1):
        """
        If only one argument is passed in, return the results of
        running the network on that vector.

        If a second argument is received, train the network to return
        that result given the first argument.
        """
        # run through the network in the forward direction
        for layer in self.layers:
            vector = layer(vector)

        if expected is not None:
            # back propogate errors and train layers
            error_out = vector - expected
            for i, layer in reversed(list(enumerate(self.layers))):
                new_error_out = layer.error_in(error_out)
                layer.train(error_out, learning_rate)
                error_out = new_error_out
            vector = (vector, error_out)

        return vector

    def save(self, dirname):
        """
        Saves all layer weights as
        """
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for i, l in enumerate(self.layers):
            d = l.weights.shape[0]
            filename = os.path.join(dirname, \
                                    'weights_{:0=2d}_{}.npy'.format(i, d))
            tofile(filename, l.weights)


def load(dirname):
    if not os.path.exists(dirname):
        return None

    weight_files = [f for f in os.listdir(dirname) if f.startswith('weights_')]
    weight_files = sorted(weight_files, key=lambda x: int(x[8:10]))
    network = Network(None)

    if len(weight_files) == 0:
        return None

    for i, filename in enumerate(weight_files):
        d = int(filename[11:-4])
        weights = fromfile(os.path.join(dirname, filename), np.float32)

        if i == 0:
            l = Layer(1, 1, 'tanh')
        elif i == len(weight_files) - 1:
            l = Layer(1, 1, 'tanh')
        else:
            l = Layer(1, 1, 'logit')
        l.weights = weights.reshape((d, weights.shape[0] // d))
        l.momentum = np.zeros_like(l.weights)
        network.layers.append(l)

    return network
