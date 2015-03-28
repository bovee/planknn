import numpy as np
from pyplanknn.network import load
import matplotlib.pyplot as plt


def plot_convs():
    conv_wgts = load('test_dump.txt').layers[0].weights

    for i, wgt in enumerate(conv_wgts):
        plt.subplot(4, 4, i)
        plt.imshow(wgt)
    plt.show()
