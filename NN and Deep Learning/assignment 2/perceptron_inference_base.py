import numpy as np
import os
import sys
from nn_load import *
from perceptron_inference_solution import perceptron_inference


# Specify parameters for a test case.
weights_file = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 2/weights/weights2.txt"
input_file = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 2/inputs/input2a.txt"

#activation_string = "step"
activation_string = "sigmoid"

# Read the weights of the perceptron.
weights = read_matrix(weights_file)
b = weights[0,0]
w = weights[1:, :]

# Read the input vector.
input_vector = read_matrix(input_file)

# The next line is where your function is called.
(a, z) = perceptron_inference(b, w, activation_string, input_vector)

# Print the results.
print("a = %.4f\nz = %.4f" % (a, z))
