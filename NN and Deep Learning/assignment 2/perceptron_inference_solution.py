# Long Nguyen
# 1001705873

import numpy as np
'''
- b: a real number (float) specifying the bias weight for the perceptron.
- w: a column vector spcifying the weights of w of the perceptron.
- activation: it is a string that specifies the activation function.
- input_vector: a column vector specifying the input to the perceptron.

- step function h(a) = {0, if a < 0; 1, if a >= 0
- sigmoid function 𝜎(a) = 1 / (1 + e^-a)
'''
def perceptron_inference(b, w, activation, input_vector):
    # check if shape a w is the same as input_vector
    if w.shape != input_vector.shape:
        raise ValueError("Mismatch: w and input_vector must have the same size.")
    
    # perform matrix multiplication to get w^T * x
    wtx = (np.transpose(w) @ input_vector).item()
    # perform step 1: a = b + w^T * x
    a = b + wtx

    # perform step 2: z = h(a) or z = 𝜎(𝑎) 
    if activation == "sigmoid":
        # sigmoid function
        z = 1 / (1 + np.exp(-a))
    else:
        # step function
        z = 0 if a < 0 else 1
    
    return a, z