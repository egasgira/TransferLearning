from __future__ import division
import numpy as np
from keras.utils import np_utils

def preprocess(x_data, y_data):
    # Normalize data
    x_data = x_data.astype('float32') / 255

    # Adapt the labels to the one-hot vector syntax required by the softmax
    size = max(y_data) + 1
    y_data = np.array([np_utils.to_categorical(i, size) for i in y_data])
    return x_data, y_data