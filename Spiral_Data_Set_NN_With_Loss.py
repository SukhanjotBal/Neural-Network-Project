import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dataset
X, y = spiral_data(samples=100, classes=3)


# Layers
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


class Softmax:
    def fwrd(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )
        self.output = exp_values / np.sum(
            exp_values, axis=1, keepdims=True
        )
        return self.output


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x):
    return x * sigmoid(x)


# Loss functions
class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        return np.mean(sample_losses)


class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis=1
            )

        return -np.log(correct_confidences)


# Initialize layers
Layer1 = Layer_Dense(2, 4)
Layer2 = Layer_Dense(4, 16)
Output = Layer_Dense(16, 3)

ReLU_Activation = ReLU()
Softmax_Activation = Softmax()
Loss_Function = Loss_CCE()


# Forward pass
Layer1_output = Layer1.fwrd(X)
Layer1_Z = ReLU_Activation.fwrd(Layer1_output)

Layer2_output = Layer2.fwrd(Layer1_Z)
Layer2_Z = swish(Layer2_output)

Output_logits = Output.fwrd(Layer2_Z)
Output_Z = Softmax_Activation.fwrd(Output_logits)


# Predictions
predictions = np.argmax(Output_Z, axis=1)
accuracy = np.mean(predictions == y)

print("Output probabilities:")
print(Output_Z)

print("\nOutput shape:", Output_Z.shape)
print("Probability sums:", np.sum(Output_Z, axis=1))

print("\nPredictions shape:", predictions.shape)
print("Accuracy:", accuracy)


# Loss
loss = Loss_Function.calculate(Output_Z, y)
print("Loss:", loss)


# Visualize results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10)
plt.title("Actual Classes")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="brg", s=10)
plt.title("Predicted Classes")

plt.tight_layout()
plt.show()
