import numpy as np
from pyplanknn.helper import outer
#TODO: implement weight constraints (Hinton lecture 9b)
# -> limit maximum squared length
#TODO: momentum-based gradients


class Layer:
    """
    A layer in a neural network that linearly transforms its input.
    """
    def __init__(self, inputs, outputs, func):
        self._input = None
        self._f_input = None

        if func == 'logit':
            self.f = lambda x: 1 / (1 + np.exp(-x))
            self.df = lambda x: self.f(x) * (1 - self.f(x))
        elif func == 'tanh':
            self.f = lambda x: np.tanh(x)
            self.df = lambda x: 1 - np.tanh(x) ** 2
        elif func == 'softmax':
            self.f = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
            self.df = lambda x: np.ones(x.shape)
        elif func == 'relu':
            self.f = lambda x: np.where(x > 0, x, 0)
            self.df = lambda x: np.where(x > 0, x, 1)
        else:
            self.f = lambda x: x
            self.df = lambda x: 1

        #self.weights = np.random.normal(0, 0.1 / np.log(inputs * outputs), \
        #                                (inputs + 1, outputs))
        self.weights = np.random.normal(0, 1 / np.sqrt(inputs * outputs), (inputs + 1, outputs))
        # deemphasize the bias terms
        self.weights[:, 0] /= 100.
        self.weights = self.weights.astype(np.float32)
        self.momentum = np.zeros_like(self.weights)

    def __call__(self, vector):
        # append 1 to beginning of data for bias term
        v_bias = np.hstack([np.ones((vector.shape[0], 1)), vector])

        self._input = v_bias
        self._f_input = np.dot(self._input, self.weights)
        return self.f(self._f_input)

    def error_in(self, error_out):
        # return the error coming into this layer, cutting out the bias term
        error_in_f = self.df(self._f_input) * error_out
        error_in = np.dot(error_in_f, self.weights.T)[:, 1:]
        return error_in

    def train(self, error_out, learning_rate):
        #TODO: use momentum? or rmsprop? Hinton lectures 6c and 6e
        error_in_f = self.df(self._f_input) * error_out

        #TODO: should divide the gradient by minibatch size?
        #http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html

        # update the weights
        gradient = np.zeros_like(self.weights)
        for i, j in zip(self._input, error_in_f):
            gradient += outer(i, j)
        gradient /= len(self._input)
        #gradient = np.mean([outer(i, j) for i, j in \
        #                    zip(self._input, error_in_f)], axis=0)

        self.momentum = 0.9 * self.momentum - learning_rate * gradient
        self.weights += self.momentum
