import os
import numpy as np
from pyplanknn.helper import fromfile
try:
    import cv2
except ImportError:
    # running in pypy after preprocessing is complete
    pass

THRESHOLD = 254
DIL_MASK = np.ones((4, 4)).astype(np.uint8)

MAX_X, MAX_Y = 250, 250

DPATH = '..'


def primary_feat(imagepath):
    im = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2GRAY)
    _, imthr = cv2.threshold(im, 250, 1, cv2.THRESH_BINARY_INV)
    imdilated = cv2.dilate(imthr, DIL_MASK)

    _, contours, hier = cv2.findContours(imdilated.copy(), cv2.RETR_TREE, \
                                         cv2.CHAIN_APPROX_SIMPLE)
    mr = max(contours, key=cv2.contourArea)

    if len(mr) < 4:
        return None

    mask = np.zeros(im.shape, np.uint8)
    cv2.drawContours(mask, [mr], 0, 255, -1)

    _, (box_w, box_h), _ = cv2.minAreaRect(mr)
    box_w, box_h = int(box_w), int(box_h)
    from_box = cv2.boxPoints(cv2.minAreaRect(mr))
    to_box = np.array([[0, 0], [box_h, 0], \
                       [box_h, box_w], [0, box_w]], np.float32)
    transform = cv2.getPerspectiveTransform(from_box, to_box)
    imrot = cv2.warpPerspective(mask * im, transform, (box_h, box_w))
    return imrot


def pad_image(image):
    blank = np.zeros((MAX_X, MAX_Y), np.uint8)

    x_corner = MAX_X // 2 - image.shape[0] // 2
    if x_corner <= 0:
        x_corner = image.shape[0] // 2 - MAX_X // 2
        from_x = slice(x_corner, x_corner + MAX_X)
        to_x = slice(None)
    else:
        from_x = slice(None)
        to_x = slice(x_corner, x_corner + image.shape[0])

    y_corner = MAX_Y // 2 - image.shape[1] // 2
    if y_corner <= 0:
        y_corner = image.shape[1] // 2 - MAX_Y // 2
        from_y = slice(y_corner, y_corner + MAX_Y)
        to_y = slice(None)
    else:
        from_y = slice(None)
        to_y = slice(y_corner, y_corner + image.shape[1])

    blank[to_x, to_y] = image[from_x, from_y]
    return blank
    #cv2.imwrite(imagepath, cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB))


def preprocess_training(training_dir, classes):
    # max train feature image size = x:537, y:529

    # precomputed
    N_FILES = 30336

    X = np.empty((N_FILES, MAX_X * MAX_Y), dtype=np.uint8)
    y = np.empty((N_FILES), dtype=np.uint8)

    file_n = 0
    for path, _, filenames in os.walk(training_dir):
        if len(filenames) == 0:
            continue
        classname = os.path.split(path)[1]
        class_n = classes.index(classname)
        for filename in filenames:
            filepath = os.path.join(path, filename)
            if filepath[-4:] != '.jpg':
                continue

            X[file_n] = pad_image(primary_feat(filepath)).flatten()
            y[file_n] = class_n
            file_n += 1

    X.tofile(os.path.join(DPATH, 'x_train.npy'))
    y.tofile(os.path.join(DPATH, 'y_train.npy'))


def preprocess_test(test_dir):
    #precomputed
    N_FILES = 130400
    X = np.empty((N_FILES, MAX_X * MAX_Y), dtype=np.uint8)

    file_n = 0
    filenames = os.listdir(test_dir)
    for filename in filenames:
        filepath = os.path.join(test_dir, filename)
        X[file_n] = pad_image(primary_feat(filepath)).flatten()
        file_n += 1

    X.tofile(os.path.join(DPATH, 'x_test.npy'))
    with open(os.path.join(DPATH, 'filenames_test.txt'), 'w') as f:
        f.write(','.join(filenames))


def read_train():
    X = fromfile(os.path.join(DPATH, 'x_train.npy'), dtype=np.uint8)
    X = np.reshape(X, (X.shape[0] // (MAX_X * MAX_Y), MAX_X, MAX_Y))

    y = fromfile(os.path.join(DPATH, 'y_train.npy'), dtype=np.uint8)

    return X, y


def read_test():
    X = fromfile(os.path.join(DPATH, 'x_test.npy'), dtype=np.uint8)
    #X = np.fromfile(os.path.join(DPATH, 'x_test.npy'), dtype=np.uint8)
    np.reshape(X, (X.shape[0] / (MAX_X * MAX_Y), MAX_X, MAX_Y))
    return X


def read_meta():
    filename_file = os.path.join(DPATH, 'test_filenames.txt')
    filenames = open(filename_file, 'r').read().strip().split(',')
    class_file = os.path.join(DPATH, 'classes.txt')
    classes = open(class_file, 'r').readline().rstrip().split(',')
    return classes, filenames
