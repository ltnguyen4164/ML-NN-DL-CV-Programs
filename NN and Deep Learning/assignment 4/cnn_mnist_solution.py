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
    training_imgs = training_imgs.astype("float32") / 255.0
    test_imgs = test_imgs.astype("float32") / 255.0
    
    return training_imgs, training_labels, test_imgs, test_labels
def create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, cnn_activation):
    # Create parameters for network
    x_train = np.expand_dims(training_inputs, -1)
    input_shape = x_train[0].shape
    num_classes = np.max(training_labels) + 1
    
    # Create CNN
    model = tf.keras.Sequential()

    # Create input layer
    model.add(tf.keras.Input(shape=input_shape))
    
    # Create hidden layer(s)
    for i in range(blocks):
        model.add(tf.keras.layers.Conv2D(filter_number, kernel_size=(filter_size, filter_size), activation=cnn_activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(region_size, region_size)))

    # Create output layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Train network
    model.fit(x_train, training_labels, epochs=epochs, verbose=0)
    
    return model