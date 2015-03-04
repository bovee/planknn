#!/usr/bin/env python3
from planknn.submit import run_model
from pyplanknn.network import load

WEIGHTS_FILE = '../../test_dump.txt'


if __name__ == '__main__':
    model = load(WEIGHTS_FILE)
    run_model(model)
