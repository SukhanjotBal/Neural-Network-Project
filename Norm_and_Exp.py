import math
import numpy as np

outputs = [0.28, 0.98, 1.51]
exp_values = np.exp(outputs) # exp function


norm_base = sum(exp_values) 
norm_values = []

for values in exp_values: # norm loop
    norm_values.append(values/norm_base)
    print(norm_values, end='')
    print(sum(norm_values))