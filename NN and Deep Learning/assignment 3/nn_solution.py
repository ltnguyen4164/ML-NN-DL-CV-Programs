# Long Nguyen
# 1001705873

import numpy as np

# Function that converts a label to its one-hot vector representation
def convert_to_one_hot(label, num_classes):
    # Set vector t dimensions to number of class
    t = np.zeros(num_classes)  
    # if label = i, then set the i-th dimension of t to 1
    # else, set the i-th dimension of t to 0
    t[label] = 1
    return t
def nn_train_and_test(tr_data, tr_labels, test_data, test_labels, labels_to_ints, ints_to_labels, parameters):
    # Set parameters to variables (for easier typing)
    L = parameters.num_layers
    J = [tr_data.shape[1]] + parameters.units_per_layer + [len(labels_to_ints)]
    R = parameters.training_rounds
    
    # Find max value for training and test data
    tr_max = np.max(tr_data)
    test_max = np.max(test_data)
    
    # Normalize training and test data using max value
    norm_tr = tr_data / tr_max
    norm_test = test_data / test_max
    
    # Define number of input features and output classes
    sets, features = tr_data.shape
    num_classes = len(labels_to_ints)
    
    # Initialize weights and bias to random values
    # Initialize the learning rate (eta) to 1
    weights = [None] + [np.random.uniform(-0.05, 0.05, size=(J[l-1], J[l])) for l in range(1, L)]
    bias = [None] + [np.random.uniform(-0.05, 0.05, size=J[l]) for l in range(1, L)]
    eta = 1
    
    # Convert labels to one-hot encoding
    tr_one_hot = np.array([convert_to_one_hot(label, num_classes) for label in tr_labels])
    
    # Training phase: ends when number of training_rounds is reached
    for r in range(R):  
        for x_n, t_n in zip(norm_tr, tr_one_hot):
            # Step 1: Initialize input layer
            z = [None] * L
            z[0] = x_n

            # Step 2: Compute outputs
            a = [None] * L
            for l in range(1, L):
                a[l] = np.dot(z[l - 1], weights[l]) + bias[l]
                z[l] = 1 / (1 + np.exp(-a[l]))
            
            # Step 3: Compute new δ values
            delta = [None] * L
            delta[L - 1] = (z[L - 1] - t_n) * z[L - 1] * (1 - z[L - 1])
            
            for l in range(L - 2, 0, -1):
                delta[l] = np.dot(weights[l + 1], delta[l + 1]) * z[l] * (1 - z[l])
            
            # Step 4: Update weights and biases
            alpha = min(1, 3000 / len(tr_data))
            for l in range(1, L):
                bias[l] -= eta * (alpha * np.mean(delta[l], axis=0) + (1 - alpha) * delta[l])
                weights[l] -= eta * np.outer(z[l - 1], delta[l])     

        # Update learning rate
        eta *= 0.98

    # Inference phase
    classification_acc = 0
    for i in range(test_data.shape[0]):
        z = [None] * L
        z[0] = norm_test[i]

        # Produce output z for each layer
        for l in range(1, L):
            a = np.dot(z[l-1], weights[l]) + bias[l]
            z[l] = 1 / (1 + np.exp(-a))

        # Predict class based on which perceptron produces highest z
        max_z = np.max(z[L-1])
        tied_classes = np.where(z[L-1] == max_z)[0]
        predicted_class = np.random.choice(tied_classes)

        # Get accuracy of predicted class
        if len(tied_classes) == 1: # No ties
            accuracy = 1 if predicted_class == test_labels[i] else 0
        else: # Ties exist
            accuracy = 1 / len(tied_classes) if test_labels[i] in tied_classes else 0
        classification_acc += accuracy

        # Print output
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (i, str(ints_to_labels[predicted_class]), str(ints_to_labels[test_labels[i, 0]]), accuracy))
        
    # Get classification accuracy
    classification_acc = classification_acc / test_data.shape[0]
    print('classification accuracy=%6.4f' % (classification_acc))