import numpy as np

np.random.seed(0)

inputs = np.array([
    [9, 0, 3],
    [4, 5, 7],
    [6, 1, 9]
])


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
        return self.output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x):
    return x * sigmoid(x)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Initialize layers
Layer1 = Layer_Dense(3, 4)
Layer2 = Layer_Dense(4, 16)
Output = Layer_Dense(16, 3)

Activation = ReLU()

# Forward pass
Layer1_output = Layer1.fwrd(inputs)
Layer1_Z = Activation.fwrd(Layer1_output)

Layer2_output = Layer2.fwrd(Layer1_Z)
Layer2_Z = swish(Layer2_output)

Output_logits = Output.fwrd(Layer2_Z)
Output_Z = softmax(Output_logits)

# Display results
print("Output probabilities:")
print(Output_Z)

print("\nOutput shape:", Output_Z.shape)
print("Probability sums:", np.sum(Output_Z, axis=1))