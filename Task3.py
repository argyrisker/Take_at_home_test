import numpy as np


def relu(x):
    return np.maximum(0, x)


def global_average_pooling(x):
    return np.mean(x, axis=(1, 2))


def relu_and_gap(x):
    x = relu(x)
    x = global_average_pooling(x)
    x = x.flatten() 

    return x