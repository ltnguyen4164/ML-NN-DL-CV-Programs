# Long Nguyen
# 1001705873

import tensorflow as tf
import numpy as np

def load_mnist():
    # Import and load mnist dataset
    dataset = tf.keras.datasets.mnist
    (training_data, test_data) = dataset.load_data()
    (training_imgs, training_labels) = training_data
    (test_imgs, test_labels) = test_data

    # Normalize input images to range [0, 1]
    # Divide imgs by mnist image max pixel value
    training_imgs = training_imgs.astype("float32") / 255.0
    test_imgs = test_imgs.astype("float32") / 255.0
    
    return training_imgs, training_labels, test_imgs, test_labels
def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    # Create parameters for network
    (idx, rows, cols) = training_inputs.shape
    num_classes = np.max([np.max(training_labels)]) + 1
    
    # Create a 2-layer network
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.layers.Flatten(input_shape=(rows, cols)))
    # Hidden layers
    if layers > 2:
        for i in range(layers - 2):
            model.add(tf.keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))
    # Output layers
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Train network
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=0)
    
    return model