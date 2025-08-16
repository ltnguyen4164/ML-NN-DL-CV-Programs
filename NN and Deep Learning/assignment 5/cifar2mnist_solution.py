# Long Nguyen
# 1001705873

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    # Compile model
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    # Train model
    model.fit(cifar_tr_inputs, cifar_tr_labels, batch_size=batch_size, epochs=epochs)
    
def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    # Resize training inputs
    training_inputs = tf.expand_dims(training_inputs, axis=-1)
    training_inputs = tf.image.grayscale_to_rgb(training_inputs)
    
    # Load pretrained model
    model = tf.keras.models.load_model('cifar10_e20_b128_6.keras')
    
    # Get the parameters for new model
    num_layers = len(model.layers)
    input_shape = training_inputs[0].shape
    num_classes = np.max(training_labels) + 1
    
    # Create new model
    refined_model = tf.keras.Sequential([tf.keras.Input(shape=input_shape)]+
                                        model.layers[0:num_layers-1]+
                                        [layers.Dense(num_classes, activation="softmax")])
    
    # Freeze pretrained layers in new model
    for i in range(0, num_layers - 1):
        refined_model.layers[i].trainable = False
        
    # Compile new model
    refined_model.compile(optimizer="adam",
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
    
    # Train new model
    refined_model.fit(training_inputs, training_labels, batch_size=batch_size, epochs=epochs)
    
    return refined_model
    
def evaluate_my_model(model, test_inputs, test_labels):
    # Resize test inputs
    test_inputs = tf.expand_dims(test_inputs, axis=-1)
    test_inputs = tf.image.grayscale_to_rgb(test_inputs)
    
    # Test model using test inputs
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    return test_acc