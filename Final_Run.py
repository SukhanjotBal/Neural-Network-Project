import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dataset
X, y = spiral_data(samples=100, classes=3)


# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def fwrd(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


# ReLU Activation
class ReLU:
    def fwrd(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


# Softmax Activation
class Softmax:
    def fwrd(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )
        self.output = exp_values / np.sum(
            exp_values, axis=1, keepdims=True
        )
        return self.output


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x):
    return x * sigmoid(x)


def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


# Loss Functions
class Loss:
    def calculate(self, output, y_true):
        return np.mean(self.forward(output, y_true))


class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]
        else:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis=1
            )

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

        return self.dinputs


# Initialize Layers
Layer1 = Layer_Dense(2, 4)
Activation1 = ReLU()

Layer2 = Layer_Dense(4, 16)

Output = Layer_Dense(16, 3)
Softmax_Activation = Softmax()
Loss_Function = Loss_CCE()


# Training Settings
learning_rate = 1.0
epochs = 10001

loss_history = []
accuracy_history = []


# Training Loop
for epoch in range(epochs):

    # FORWARD PASS
    Layer1_output = Layer1.fwrd(X)
    Layer1_Z = Activation1.fwrd(Layer1_output)

    Layer2_output = Layer2.fwrd(Layer1_Z)
    Layer2_Z = swish(Layer2_output)

    Output_logits = Output.fwrd(Layer2_Z)
    Output_Z = Softmax_Activation.fwrd(Output_logits)

    # Calculate Loss and Accuracy
    loss = Loss_Function.calculate(Output_Z, y)
    predictions = np.argmax(Output_Z, axis=1)
    accuracy = np.mean(predictions == y)

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    # BACKPROPAGATION

    # Softmax + Cross-Entropy gradient
    dvalues = Loss_Function.backward(Output_Z, y)

    # Output Layer
    dvalues = Output.backward(dvalues)

    # Swish derivative
    dvalues *= swish_derivative(Layer2_output)

    # Hidden Layer 2
    dvalues = Layer2.backward(dvalues)

    # ReLU derivative
    dvalues = Activation1.backward(dvalues)

    # Hidden Layer 1
    Layer1.backward(dvalues)

    # GRADIENT DESCENT
    Layer1.update(learning_rate)
    Layer2.update(learning_rate)
    Output.update(learning_rate)

    # Progress
    if epoch % 1000 == 0:
        print(
            f"Epoch: {epoch:5d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


# Final Results
print("\nFinal Loss:", loss_history[-1])
print("Final Accuracy:", accuracy_history[-1])


# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10)
axes[0].set_title("Actual Classes")

axes[1].scatter(
    X[:, 0], X[:, 1], c=predictions, cmap="brg", s=10
)
axes[1].set_title("Predicted Classes")

axes[2].plot(loss_history)
axes[2].set_title("Training Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Cross-Entropy Loss")

plt.tight_layout()
plt.show()
