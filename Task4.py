import os
import numpy as np


def load_fc(folder):
    wfc_path = os.path.join(folder, "fc_weight.csv")
    bfc_path = os.path.join(folder, "fc_bias.csv")

    wfc = np.loadtxt(wfc_path, delimiter=",", dtype=np.float32)
    bfc = float(np.loadtxt(bfc_path, delimiter=",", dtype=np.float32))

    wfc = wfc.reshape(-1)

    return wfc, bfc


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def classify_vector(o, wfc, bfc):
    s = bfc + np.dot(o, wfc)
    p = sigmoid(s)

    return float(p)