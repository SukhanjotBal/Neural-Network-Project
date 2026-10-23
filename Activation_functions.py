import numpy as np
class ReLU:
    def fwrd(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
    
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x * sigmoid(x)