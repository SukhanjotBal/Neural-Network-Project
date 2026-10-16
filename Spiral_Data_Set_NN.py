import numpy as np
import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)


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
Layer1 = Layer_Dense(2, 4)
Layer2 = Layer_Dense(4, 16)
Output = Layer_Dense(16, 3)

Activation = ReLU()

# Forward pass
Layer1_output = Layer1.fwrd(X)
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

predictions = np.argmax(Output_Z, axis=1)

accuracy = np.mean(predictions == y)

print("\nPredictions shape:", predictions.shape)
print("Accuracy:", accuracy)

# Visualize the dataset and predictions
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10)
plt.title("Actual classes")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="brg", s=10)
plt.title("Predicted classes")

plt.tight_layout()
plt.show()
