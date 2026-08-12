import numpy as np

def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x * sigmoid(x)