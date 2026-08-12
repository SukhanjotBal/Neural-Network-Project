import numpy as np

inputs = np.array([9, 0, 3])

weight1 = np.array([3, 4, 2])
weight2 = np.array([8, 0, 2])
weight3 = np.array([4, 8, 4])

bias1 = 2
bias2 = 4
bias3 = 8

output1 = np.dot(inputs, weight1) + bias1
output2 = np.dot(inputs, weight2) + bias2
output3 = np.dot(inputs, weight3) + bias3

print(f"output1 is {output1}, output2 is {output2} and output3 is {output3}")