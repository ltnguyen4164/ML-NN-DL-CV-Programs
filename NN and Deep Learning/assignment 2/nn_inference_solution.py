# Long Nguyen
# 1001705873

import numpy as np
'''
- layers, units, biases, weights: arguments that specify the number of
layers, number of units per layer, and the weights b and w of each unit
in each layer
- activation: a string that specifies the activation function
- input_vector: a column vector specifying the input to the perceptron
'''
def nn_inference(layers, units, biases, weights, activation, input_vector):
    # initialize a_values and z_values
    a_values = [None, None]
    z_values = [None, input_vector]

    # define activation functions (sigmoid or step)
    if activation == "sigmoid":
        def activation_func(a):
           return [1 / (1 + np.exp(-val)) for val in a]
    elif activation == "step":
        def activation_func(a):
            return [0 if val < 0 else 1 for val in a]
            
    # iterate through layers skipping the input layer
    for layer in range(2, layers + 1):
        # get weight(s) and bias(es) for this layer
        w = weights[layer]
        b = biases[layer]

        # initialize empty lists to accumulate a and z values for this layer
        a_val = []
        z_val = []
        # only the layer after the input layer uses the input vector
        if layer == 2:
            if w is not None and b is not None:
                if w.shape[0] > 1:
                    for arr in range(w.shape[0]):    
                        # perform step 1: a = b + w^T * x
                        wtx = (np.transpose(w[arr]) @ input_vector) 
                        a = wtx + b[arr]
                        # perform step 2: z = h(a) or z = 𝜎(𝑎) 
                        z = activation_func(a)
                        # append computed values
                        a_val.append(a)
                        z_val.append(z)
                else:
                    # perform step 1: a = b + w^T * x
                    wtx = (w @ input_vector) 
                    a = wtx + b
                    # perform step 2: z = h(a) or z = 𝜎(𝑎) 
                    z = activation_func(a)
                    # append computed values
                    a_val.append(a)
                    z_val.append(z)
                a_values.append(np.array(a_val))
                z_values.append(np.array(z_val))
        else:
            if w is not None and b is not None:
                if w.shape[0] > 1:
                    for arr in range(w.shape[0]):    
                        # perform step 1: a = b + w^T * x
                        wtx = (np.transpose(w[arr]) @ z_values[layer - 1]) 
                        a = wtx + b[arr]
                        # perform step 2: z = h(a) or z = 𝜎(𝑎) 
                        z = activation_func(a)
                        # append computed values
                        a_val.append(a)
                        z_val.append(z)
                else:
                    # perform step 1: a = b + w^T * x
                    wtx = (w @ z_values[layer - 1]) 
                    a = wtx + b
                    # perform step 2: z = h(a) or z = 𝜎(𝑎) 
                    z = activation_func(a)
                    # append computed values
                    a_val.append(a)
                    z_val.append(z)
                a_values.append(np.array(a_val))
                z_values.append(np.array(z_val))
        
    # reshape a and z values to fit nn_load
    a_values = [None, None] + [arr.reshape(-1, 1) if arr is not None else None for arr in a_values[2:]]
    z_values = [None, input_vector] + [arr.reshape(-1, 1) if arr is not None else None for arr in z_values[2:]]
    return a_values, z_values    