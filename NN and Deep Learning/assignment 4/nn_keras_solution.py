# Long Nguyen
# 1001705873

import tensorflow as tf
import numpy as np

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    # Create parameters for network
    input_shape = training_inputs[0].shape
    num_classes = np.max([np.max(training_labels)]) + 1
    
    # Create a 2-layer network
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.Input(shape=input_shape))
    # Hidden layers
    for i in range(layers - 2):
        model.add(tf.keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))
    # Output layers
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Who is adam, and why is he optimizing my model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Train network
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=0)
    
    return model
def test_model(model, test_inputs, test_labels, ints_to_labels):
    test_accuracy = 0
    
    # Cycle through each test input
    for i in range(test_inputs.shape[0]):
        # Predict class with model
        input_vector = test_inputs[i, :]
        input_vector = np.expand_dims(input_vector, axis=0)
        nn_output = model.predict(input_vector, verbose=0)
        nn_output = nn_output.flatten()
        predicted_cl = np.argmax(nn_output)
        
        # Find tied classes (if any)
        max_prob = np.max(nn_output)
        tied_cl = np.where(nn_output == max_prob)[0]
        
        # Determine accuracy
        if len(tied_cl) == 1:
            # No ties
            accuracy = 1 if predicted_cl == test_labels[i] else 0
        else:
            # Ties exist
            if test_labels[i] in tied_cl:
                accuracy = 1 / len(tied_cl)
            else:
                accuracy = 0
        test_accuracy += accuracy
        
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (i, ints_to_labels[predicted_cl], ints_to_labels[test_labels[i, 0]], accuracy))
    
    return test_accuracy / test_inputs.shape[0]