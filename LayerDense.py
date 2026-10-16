import numpy as np
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def fwrd(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
class ReLU:
    def fwrd(self, inputs):
        self.output = np.maximum(0, inputs)
        return self. output
    
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x * sigmoid(x)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)