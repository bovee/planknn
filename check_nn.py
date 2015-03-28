## this may crash due to memory demands (training too many features at once)
#from pyplanknn.network import Network
#from pyplanknn.submit import check_model
#
#check_model(Network)

import random
import numpy as np
from sklearn.metrics import classification_report
from pyplanknn.preprocess import read_train, read_meta
from pyplanknn.network import load
from pyplanknn.monitoring import multiclass_log_loss

WEIGHTS_DIR = 'network_001'


if __name__ == '__main__':
    model = load(WEIGHTS_DIR)
    x, x_class = read_train()
    y = np.zeros((x_class.shape[0], 121))
    for i, c in enumerate(x_class):
        y[i, c] = 1.

    classes, _ = read_meta()

    s = random.sample(range(len(x_class)), 1000)
    x_train = (x[s] - 127.5).reshape(-1, 62500) / 127.5
    x_guess = model(x_train)

    print(multiclass_log_loss(x_class[s], x_guess))
    print(classification_report(x_class[s], x_guess.argmax(axis=1), \
                                target_names=classes))
