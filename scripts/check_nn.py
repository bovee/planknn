## this may crash due to memory demands (training too many features at once)
#from pyplanknn.network import Network
#from pyplanknn.submit import check_model
#
#check_model(Network)

import random
from sklearn.metrics import classification_report
from pyplanknn.preprocess import read_train
from pyplanknn.network import load
from pyplanknn.monitoring import multiclass_log_loss

WEIGHTS_DIR = 'network_001'


if __name__ == '__main__':
    model = load(WEIGHTS_DIR)
    x, x_class = read_train()

    s = random.sample(range(len(x_class)), 1000)
    x_guess = model(x[s] / 255)
    print(multiclass_log_loss(x_class[s], x_guess))
    print(classification_report(x_class[s], x_guess))
