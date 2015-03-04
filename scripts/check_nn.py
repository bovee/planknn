## this may crash due to memory demands (training too many features at once)
#from pyplanknn.network import Network
#from pyplanknn.submit import check_model
#
#check_model(Network)

import random
from pyplanknn.preprocess import read_train
from pyplanknn.network import load
from pyplanknn.monitoring import multiclass_log_loss

WEIGHTS_FILE = '../../test_dump.txt'


if __name__ == '__main__':
    model = load(WEIGHTS_FILE)
    x, x_class = read_train()

    s = random.sample(range(len(x_class)), 100)
    print(multiclass_log_loss(x_class[s], model(x[s] / 255)))
