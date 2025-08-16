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
def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels, labels_to_ints, ints_to_labels, training_rounds):
    # Find max value for training and test data
    tr_max = np.max(tr_data)
    test_max = np.max(test_data)
    
    # Normalize training and test data using max value
    norm_tr = tr_data / tr_max
    norm_test = test_data / test_max
    
    # Get number of class
    num_classes = len(labels_to_ints)
    
    # Initialize weights and bias to random values
    # Initialize the learning rate (eta) to 1
    sets, features = tr_data.shape
    weights = np.random.uniform(-0.05, 0.05, size=(num_classes, features))
    bias = np.random.uniform(-0.05, 0.05, size=num_classes)
    eta = 1
    
    # Training phase: ends when number of training_rounds is reached
    for r in range(training_rounds):
        # Train K perceptrons, one for each class
        for k in range(num_classes):   
            for i in range(sets):
                # Convert each label to a one-hot vector
                tr_one_hot = convert_to_one_hot(tr_labels[i], num_classes)
                
                # Compute z
                wtx = (np.transpose(weights[k]) @ norm_tr[i])
                a = bias[k] + wtx
                z = 1 / (1 + np.exp(-a))
            
                # Compute gradients
                grad_b = (z - tr_one_hot[k]) * z * (1 - z)
                grad_w = (z - tr_one_hot[k]) * z * (1 - z) * norm_tr[i]
                
                # Update weights and bias
                bias[k] -= eta * grad_b
                weights[k] -= eta * grad_w

        # Update learning rate
        eta *= 0.98

    # Inference phase
    classification_acc = 0
    for i in range(test_data.shape[0]):
        z_values = []

        # Produce output z for each perceptron k
        for k in range(num_classes):
            # Compute z
            wtx = (np.transpose(weights[k]) @ norm_test[i])
            a = bias[k] + wtx
            z = 1 / (1 + np.exp(-a))
            # Store z value for perceptron k
            z_values.append(z)

        # Predict class based on which perceptron produces highest z
        max_z = np.max(z_values)
        tied_classes = np.where(z_values == max_z)[0]
        predicted_class = np.random.choice(tied_classes)

        # Get accuracy of predicted class
        if len(tied_classes) == 1: # No ties
            accuracy = 1 if predicted_class == test_labels[i] else 0
        else: # Ties exist
            if test_labels[i] in tied_classes:
                accuracy = 1 / len(tied_classes)
            else:
                accuracy = 0
        classification_acc += accuracy

        # Print output
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (i, str(ints_to_labels[predicted_class]), str(ints_to_labels[test_labels[i, 0]]), accuracy))
        
    # Get classification accuracy
    classification_acc = classification_acc / test_data.shape[0]
    print('classification accuracy=%6.4f' % (classification_acc))