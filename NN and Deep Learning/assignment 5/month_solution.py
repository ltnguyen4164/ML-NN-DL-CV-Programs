# Long Nguyen
# 1001705873

import numpy as np
import tensorflow as tf

def data_normalization(raw_data, train_start, train_end):
    # Extract training segment
    train_data = raw_data[train_start:train_end]
    
    # Compute mean and standard deviation for each feature
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    # Avoid division by zero by replacing zero std with 1
    std[std == 0] = 1
    
    # Normalize the entire dataset
    normalized_data = (raw_data - mean) / std
    
    return normalized_data

def make_inputs_and_targets(data, months, size, sampling):
    # Function from timeseries_code.py with adjustments
    def random_input(timeseries, length, target_time, target_data, sampling_rate):
        (ts_length, dimensions) = timeseries.shape
        max_start = ts_length - (length * sampling) - target_time # Adjust for sampling rate > 1
        start = np.random.randint(0, max_start)
        end = start + (length * sampling) # Adjust for sampling rate > 1
        result_input = timeseries[start:end:sampling_rate, :]
        target = target_data[end + target_time - sampling_rate]
        return (result_input, target, start)
    
    # 2,016 = Number of observations per 10 minutes in a 2 week time period
    timesteps = 2016 // sampling 
    num_features = data.shape[1] # Number of features per observation
    
    inputs = np.zeros((size, timesteps, num_features))
    targets = np.zeros(size, dtype=int)

    for i in range(size):
        inp, target, _ = random_input(data, timesteps, timesteps // 2, months, sampling)
        inputs[i] = inp
        targets[i] = target
    
    return inputs, targets   

def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    # Get parameters for model
    input_shape = train_inputs.shape[1:]
    
    # Create model
    model = tf.keras.Sequential([tf.keras.Input(shape=input_shape),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(64, activation="tanh"),
                                 tf.keras.layers.Dense(32, activation="relu"),
                                 tf.keras.layers.Dense(12, activation="softmax"), # 12 output classes for months
                                 ])

    # Compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    # Create callback to save the best model in the filename
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filename, save_best_only=True, monitor="val_accuracy", mode="max")]

    # Train model and save the training history
    history = model.fit(train_inputs, train_targets, epochs=10,
                        validation_data=(val_inputs, val_targets),
                        callbacks=callbacks)
    return history

def test_model(filename, test_inputs, test_targets):
    # Load model
    model = tf.keras.models.load_model(filename)

    # Evaluate model
    (loss, test_mae) = model.evaluate(test_inputs, test_targets, verbose=0)

    return test_mae

def confusion_matrix(filename, test_inputs, test_targets):
    # Load model
    model = tf.keras.models.load_model(filename)
    predicts = model.predict(test_inputs)
    predicted_cl = np.argmax(predicts, axis=1)

    num_classes = 12
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate conf matrix
    for actual, predicted in zip(test_targets, predicted_cl):
        conf_matrix[actual, predicted] += 1

    return conf_matrix