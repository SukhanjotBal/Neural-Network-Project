import numpy as np

inputs = np.array([9, 0, 3])

weights = np.array([
    [3, 4, 2],
    [2, 0, 2],
])

biases = np.array([2, 4])

Z_Output = np.dot(weights, inputs) + biases

print(Z_Output)