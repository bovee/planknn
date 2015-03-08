import json
import numpy as np
from pyplanknn.layer import Layer
from pyplanknn.convlayer import ConvPoolLayer


class ConvNetwork:
    """
    A neural network comprised of several layers interconnected.
    """
    def __init__(self, n_convs=10, hidden=4, n_outs=121):
        """
        Create a series of layers.
        """
        if isinstance(hidden, int):
            args = hidden * [400] + [n_outs]
        else:
            args = hidden + [n_outs]

        args = iter(args)
        prev = next(args)
        self.layers = [ConvPoolLayer(n_convs)]
        self.layers.append(Layer(25 * n_convs, prev, 'tanh'))
        for i in args:
            self.layers.append(Layer(prev, i, 'logit'))
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
            for layer in reversed(self.layers):
                new_error_out = layer.error_in(error_out)
                layer.train(error_out, learning_rate)
                error_out = new_error_out

        return vector

    def save(self, filename):
        """
        Saves all layer weights as
        """
        weights = {}
        for i, l in enumerate(self.layers):
            weights[i] = l.weights.tolist()
        open(filename, 'w').write(json.dumps(weights))


def load(filename):
    weights = json.loads(open(filename, 'r').read())
    n_layers = max(int(i) for i in weights.keys())
    n_convs = np.array(weights['0']).shape[0]
    n_outs = np.array(weights[str(n_layers)]).shape[1]
    hidden = [np.array(weights[str(w)]).shape[1] for w \
              in range(1, n_layers - 1)]
    network = ConvNetwork(n_convs, hidden, n_outs)
    for i in range(n_layers + 1):
        network.layers[i].weights = np.array(weights[str(i)])
    return network
