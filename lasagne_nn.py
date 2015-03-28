import lasagne
from lasagne.nonlinearities import softmax, tanh

NUM_EPOCHS = 500
BATCH_SIZE = 600
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def build_network():
    l = lasagne.layers.InputLayer(shape=(121, 62500))
    for i in range(4):
        l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=tanh)
        l = lasagne.layers.DropoutLayer(l, p=0.5)
    l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=tanh)
    return lasagne.layers.DenseLayer(l, num_units=121, nonlinearity=softmax)



