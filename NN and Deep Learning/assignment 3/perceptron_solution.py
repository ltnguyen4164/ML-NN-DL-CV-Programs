# Long Nguyen
# 1001705873

import numpy as np

def perceptron_train_and_test(tr_data, tr_labels, test_data, test_labels, training_rounds):
    # Find max value for training and test data
    tr_max = np.max(tr_data)
    test_max = np.max(test_data)

    # Normalize training and test data using max value
    norm_tr = tr_data / tr_max
    norm_test = test_data / test_max

    # Initialize weights and bias to random values
    # Initialize the learning rate (eta) to 1
    sets, features = tr_data.shape
    weights = np.random.uniform(-0.05, 0.05, size=features)
    bias = np.random.uniform(-0.05, 0.05)
    eta = 1
    
    # Training phase: ends when number of trainin_rounds is reached
    for r in range(training_rounds):
        for i in range(sets):
            # Compute z
            wtx = (np.transpose(weights) @ norm_tr[i])
            a = bias + wtx
            z = 1 / (1 + np.exp(-a))
        
            # Compute gradients
            grad_b = (z - tr_labels[i]) * z * (1 - z)
            grad_w = (z - tr_labels[i]) * z * (1 - z) * norm_tr[i]
            
            # Update weights and bias
            bias -= eta * grad_b
            weights -= eta * grad_w

        # Update learning rate
        eta *= 0.98

    # Classification phase
    classification_acc = 0
    for i in range(test_data.shape[0]):
        # Compute z
        wtx = (np.transpose(weights) @ norm_test[i])
        a = bias + wtx
        z = 1 / (1 + np.exp(-a))

        # Predict class based on output z
        predicted_class = 0 if z < 0.5 else 1

        # Get accuracy of predicted class
        accuracy = 1 if predicted_class == test_labels[i] else 0
        classification_acc += accuracy

        # Print output
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (i, str(predicted_class), str(test_labels[i]), accuracy))
        
    # Get classification accuracy
    classification_acc = classification_acc / test_data.shape[0]
    print('classification accuracy=%6.4f' % (classification_acc))