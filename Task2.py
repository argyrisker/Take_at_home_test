import numpy as np
import os

def load_filter(weight, bias):
    #load weight
    w = np.loadtxt(weight, delimiter=",", dtype=np.float32)
    w = w.reshape(3, 9, 9)

    #load bias
    b = float(np.loadtxt(bias, delimiter=",", dtype=np.float32))

    return w, b



def load_all_filters(folder):
    weights = []
    biases = []

    for i in range(16):
        #get paths
        weight = os.path.join(folder, f"weight{i}.csv")
        bias = os.path.join(folder, f"bias{i}.csv")

        w,b = load_filter(weight, bias)

        weights.append(w)
        biases.append(b)

    weights = np.stack(weights, axis=0)
    biases = np.array(biases, dtype=np.float32)

    return weights, biases


def filter_image(x, weights, biases):

    padded = np.pad(x, pad_width=((0, 0), (4, 4), (4, 4)), mode="constant", constant_values=0)

    output = np.zeros((16, 64, 64), dtype=np.float32)

    for k in range(16):
        for i in range(64):
            for j in range(64):
                patch = padded[:, i:i+9, j:j+9]   
                output[k, i, j] = biases[k] + np.sum(patch * weights[k])

    return output
