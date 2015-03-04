#!/usr/bin/env python3
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from planknn.submit import run_model, check_model
from planknn.preprocess import read_train


class RFModel:
    def __init__(self, estimators=70, jobs=3):
        self.clf = RF(n_estimators=estimators, n_jobs=jobs)

    def __call__(self, vector, expected=None):
        vector = vector.reshape((vector.shape[0], -1))
        if expected is None:
            self.clf.predict_proba(vector)
        else:
            self.clf.fit(vector, expected)


if __name__ == '__main__':
    check_model(RFModel)

    # make results suitable for submission to kaggle
    model = RFModel()

    x_train, x_class = read_train()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_train[np.logical_or(np.isinf(x_train), np.isnan(x_train))] = 0
    y_train = np.zeros((x_class.shape[0], 121))
    for i, c in enumerate(x_class):
        y_train[i, c] = 1.
    model(x_train, y_train)

    run_model(model)
