from planknn.preprocess import preprocess_training, preprocess_test, read_meta

TRAINING_DIR = '../../train/'
FINAL_DIR = '../../test/'

classes, _ = read_meta()

preprocess_training(TRAINING_DIR, classes)
preprocess_test(FINAL_DIR)
