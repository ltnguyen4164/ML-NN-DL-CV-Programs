# Long Nguyen
# 1001705873

from naive_bayes import *

dataset_directory = "/Users/ltngu/Documents/CSE Files/CSE 4309/assignment 2"

dataset = "pendigits"
#dataset = "satellite"
#dataset = "yeast"

training_file = dataset_directory + "/" + dataset + "_training.txt"
test_file = dataset_directory + "/" + dataset + "_test.txt"

naive_bayes(training_file, test_file)

