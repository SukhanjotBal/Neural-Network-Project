import numpy as np

inputs = np.array([9, 0, 3])

weights = np.array([
    [3, 4, 2],
    [2, 0, 2],
])

biases = np.array([2, 4])

weights_2 = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

bias_2 = np.array([1, 2, 3])


class Layer_1:
    def __init__(self, inputs, weights, biases):
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        
    def swish(self, x, beta=1.0):
        return x / (1.0 + np.exp(-beta * x))

    def feed_frwd(self):
        Z_Output = np.dot(self.weights, self.inputs) + self.biases
        output = self.swish(Z_Output)
        return output


class Layer_2:
    def __init__(self, input_2, weights_2, bias_2):
        self.input_2 = input_2
        self.weight_2 = weights_2
        self.bias_2 = bias_2
        
    def softmax(self, x):
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)

    def feed_frwd(self):
        Z_Output_Output_Layer = (
            np.dot(self.weight_2, self.input_2) + self.bias_2
        )
        output_2 = self.softmax(Z_Output_Output_Layer)
        return output_2


# Layer 1
Layer1 = Layer_1(inputs, weights, biases)
Layer_1_output = Layer1.feed_frwd()

# Layer 2
Layer2 = Layer_2(Layer_1_output, weights_2, bias_2)
Layer_2_output = Layer2.feed_frwd()

print(f"Layer 1 output: {Layer_1_output}")
print(f"Final output: {Layer_2_output}")