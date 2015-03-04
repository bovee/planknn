import os
import numpy as np
from pyplanknn.preprocess import read_train, read_test, read_meta, DPATH
from pyplanknn.monitoring import multiclass_log_loss


def run_model(model):
    import pandas as pd
    from subprocess import check_call

    x_test = read_test()

    classes, filenames = read_meta()

    y_pred = model(x_test)

    df = pd.DataFrame(y_pred, columns=classes, index=filenames)
    df.index.name = 'image'
    df.to_csv(os.path.join(DPATH, 'submission.csv'))

    check_call(['gzip', os.path.join(DPATH, 'submission.csv')])


def check_model(Model):
    from sklearn.cross_validation import StratifiedKFold as KFold
    from sklearn.metrics import classification_report

    x, x_class = read_train()
    y = np.zeros((x_class.shape[0], 121))
    for i, c in enumerate(x_class):
        y[i, c] = 1.

    classes, _ = read_meta()

    kf = KFold(x_class, n_folds=5)
    y_pred = np.zeros(x.shape[0])
    y_pred2 = np.zeros((x.shape[0], len(classes)))
    i = 0
    for train, test in kf:
        model = Model()
        x_train, x_test = x[train, :], x[test, :]
        y_train, y_test = y[train], y[test]
        print('Training ', i)
        model(x_train, y_train)
        print('Predicting ', i)
        y_pred2[test] = model(x_test)
        y_pred[test] = y_pred2[test].argmax(axis=0)
        i += 1

    print(classification_report(y, y_pred, target_names=classes))
    print(multiclass_log_loss(y, y_pred2))
    return y_pred, y_pred2
