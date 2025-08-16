import numpy as np
from uci_load import read_uci_file
from nn_solution import nn_train_and_test

class hyperparameters:
    num_layers = None       # should be an integer >= 2
    units_per_layer = None  # should be a list of the number of units in the hidden layers.
                            # The length of this list should be num_layers - 2.
    training_rounds = None  # specifies the number of training rounds.


# Feel free to change the default directory to point to where you have
# stored the datasets.
directory = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 3/uci_data"

# Feel free to change the dataset name, so that you can experiment
# with different datasets.
#dataset = "pendigits"
#dataset = "satellite"
dataset = "yeast"

# Feel free to change the values in the next lines, so that you can experiment
# with different parameters.
parameters = hyperparameters()
parameters.num_layers = 4
parameters.units_per_layer = [20, 15]
parameters.training_rounds = 20

training_file = directory + "/" + dataset + "_training.txt"
test_file = directory + "/" + dataset + "_test.txt"

labels_to_ints = {}
ints_to_labels = {}

# These lines read the dataset
(tr_data, tr_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
(test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)

# This is where your code is called.
nn_train_and_test(tr_data, tr_labels, test_data, test_labels,
                  labels_to_ints, ints_to_labels, parameters)
