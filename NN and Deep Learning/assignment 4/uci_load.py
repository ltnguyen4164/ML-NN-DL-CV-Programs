import numpy as np
import os
import sys


# arguments:
#    - pathname: source file of the data
#    - labels_to_ints: a dictionary that maps original class labels (which 
#           can be ints or strings) to consecutive ints starting at 0
#           Note that the function MODIFIES this argument to store new
#           mappings that it creates while reading the file.
#    - ints_to_labels: a dictionary that maps int labels to original class 
#           labels (which can be ints or strings).
#           Note that the function MODIFIES this argument to store new
#           mappings that it creates while reading the file.
# returns:
#    - data: a 2D numpy array, where each row is an input.
#    - labels: a numpy column vector. That means it is a 2D numpy array, 
#           with a single column. labels[i,0] is the class label for
#           for the object stored at data[i].
def read_uci_file(pathname, labels_to_ints, ints_to_labels):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    file_lines = in_file.readlines()
    in_file.close()

    rows = len(file_lines)
    if (rows == 0):
        print("read_data: zero rows in %s", pathname)
        return None
        
    
    cols = len(file_lines[0].split())
    data = np.zeros((rows, cols-1))
    labels = np.zeros((rows,1))
    for row in range(0, rows):
        line = file_lines[row].strip()
        items = line.split()
        if (len(items) != cols):
            print("read_data: Line %d, %d columns expected, %d columns found" %(row, cols, len(items)))
            return None
        for col in range(0, cols-1):
            data[row][col] = float(items[col])
        
        # the last column is a string representing the class label
        label = items[cols-1]
        if (label in labels_to_ints):
            ilabel = labels_to_ints[label]
        else:
            ilabel = len(labels_to_ints)
            labels_to_ints[label] = ilabel
            ints_to_labels[ilabel] = label
        
        labels[row] = ilabel

    labels = labels.astype(int)
    return (data, labels)


# arguments:
#    - directory: the pathname of the folder where the dataset is stored.
#    - dataset_name: the name of the dataset, such as "pendigits" or "yeast".
# returns a tuple of three items, where each item is itself a pair, so 
# overall the function returns six values.
#    ((train_data, train_labels), (test_data, test_labels), (ints_to_labels, labels_to_ints))
#    - train_data: a 2D numpy array, where each row is a training input object.
#    - train_labels: a numpy column vector. That means it is a 2D numpy array, 
#           with a single column. train_labels[i,0] is the class label for
#           the object stored at train_data[i].
#    - test_data: a 2D numpy array, where each row is a test input object.
#    - test_labels: a numpy column vector. That means it is a 2D numpy array, 
#           with a single column. test_labels[i,0] is the class label for
#           the object stored at test_data[i].
#    - labels_to_ints: a dictionary that maps original class labels (which 
#           can be ints or strings) to consecutive ints starting at 0
#    - ints_to_labels: a dictionary that maps int labels to original class 
#           labels (which can be ints or strings).
def read_uci_dataset(directory, dataset_name):
    training_file = directory + "/" + dataset_name + "_training.txt"
    test_file = directory + "/" + dataset_name + "_test.txt"

    labels_to_ints = {}
    ints_to_labels = {}

    (train_data, train_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
    (test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)
    return ((train_data, train_labels), (test_data, test_labels), (ints_to_labels, labels_to_ints))


