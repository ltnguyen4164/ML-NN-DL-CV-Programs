from tensorflow import keras
import numpy as np

from uci_load import *
from nn_keras_solution import *


#%%

# loading the dataset

# Feel free to change the default directory to point to where you have
# stored the datasets.
#directory = "../synthetic"
directory = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 4/uci_data"
#directory = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 4"


# Feel free to change the dataset name, so that you can experiment
# with different datasets.
dataset = "synth5"
#dataset = "yeast_string"
#dataset = "ltn"
training_file = directory + "/" + dataset + "_training.txt"
test_file = directory + "/" + dataset + "_test.txt"

labels_to_ints = {}
ints_to_labels = {}

# These lines read the dataset
training_set = read_uci_file(training_file, labels_to_ints, ints_to_labels)
(training_inputs, training_labels) = training_set
test_set = read_uci_file(test_file, labels_to_ints, ints_to_labels)
(test_inputs, test_labels) = test_set

# We normalize the input values
max_value = np.max(np.abs(training_inputs))
training_inputs  = training_inputs / max_value
test_inputs = test_inputs/ max_value


# Creating the model

layers = 4
units_per_layer = [20,10]
epochs = 80
hidden_activations = ['tanh', 'sigmoid']


# here is where your create_and_train_model function is called
model = create_and_train_model(training_inputs, training_labels, layers, 
                               units_per_layer, epochs, hidden_activations)

#test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)
#print('\nTest accuracy: %.2f%%' % (test_acc * 100))

#%%

# Testing the model

# here is where your test_model function is called
test_accuracy = test_model(model, test_inputs,  test_labels, ints_to_labels)
print('Classification accuracy on test set: %.2f%%' % (test_accuracy * 100))

