import numpy as np
from uci_load import read_uci_file
from perceptron_solution import perceptron_train_and_test


# Feel free to change the default directory to point to where you have
# stored the datasets.
tr_directory = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 3/training"
test_directory = "/Users/ltngu/Documents/CSE Files/CSE 4311/assignment 3/tests"

# Feel free to change the dataset name, so that you can experiment
# with different datasets.
dataset = "synth3"

# Feel free to change the value in the next line, so that you can experiment
# with different parameters.
training_rounds = 5

training_file = tr_directory + "/" + dataset + "_training.txt"
test_file = test_directory + "/" + dataset + "_test.txt"

labels_to_ints = {}
ints_to_labels = {}

# These lines read the dataset
(tr_data, tr_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
(test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)

# This is where your code is called.
perceptron_train_and_test(tr_data, tr_labels, test_data, test_labels, 
                          training_rounds)



