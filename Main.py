import numpy as np

class Neuron:
    def __init__(self, input, weight, bias): # makes values plus __init__
        self.input = input
        self.weight = weight
        self.bias = bias
    def sigmoid(self, x): # sigmoid function and stuff, rest of the explanation of why used in commit message
        return 1 / (1 - np.exp(-x))
    def forward(self):
        z = np.dot(self.input, self.weight) + self.bias # forward function for output
        output = self.sigmoid(z)
        return output 
    
Inputs = np.array([2, 8, 5])
Weights = np.array([7, 4, 9])
Bias = 2

neuron = Neuron(Inputs, Weights, Bias) # Creates object for every variable thing idk what they're called
print("Output: ", neuron.forward()) # performs forward function for every variable thing and prints out output