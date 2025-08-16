# Long Nguyen
# 1001705873

import numpy as np
import random as rd
from collections import Counter

def parse_input(value):
    # translation dictionary for strings
    # assuming that the max number of classes is 10
    translation_dict = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
        "class0": 0.0,
        "class1": 1.0,
        "class2": 2.0,
        "class3": 3.0,
        "class4": 4.0,
        "class5": 5.0,
        "class6": 6.0,
        "class7": 7.0,
        "class8": 8.0,
        "class9": 9.0,
        "class10": 10.0
        }
    
    try:
        return float(value)
    except ValueError:
        return translation_dict.get(value, value)
def knn_classify(training_file, test_file, k):
    training_data = []
    test_data = []

    # try to open training file and extract data
    try:
        with open(training_file) as file:
            for line in file:
                training_list = [parse_input(value) for value in line.split()]
                training_data.append(training_list)
    except FileNotFoundError:
        print(f"Error: The file '{training_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()
        
    training_data = np.array(training_data)
    training_attr = training_data[:, :-1]
    training_label = training_data[:, -1]

    # try to open testing file and extract data
    try:
        with open(test_file) as file:
            for line in file:
                test_list = [parse_input(value) for value in line.split()]
                test_data.append(test_list)
    except FileNotFoundError:
        print(f"Error: The file '{test_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()
        
    test_data = np.array(test_data)
    test_attr = test_data[:, :-1]
    test_label = test_data[:, -1]

    # compute mean and std
    mean = np.mean(training_attr, axis=0)
    std = np.std(training_attr, axis=0, ddof=1)

    # handle std == 0
    std[std == 0] = 1

    # normalize training and test data
    training_attr = (training_attr - mean) / std
    test_attr = (test_attr - mean) / std
    
    # function to compute Euclidean distance
    def euclidean(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    total = 0
    object_id = 1
    distance = []
    for test_point, true_class in zip(test_attr, test_label):
        # find distance from test data point to training data point
        # used trick from https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
        # since the function above is too slow
        euclidean_distance = np.linalg.norm(training_attr - test_point, axis=1)
        # store distance with associated label
        distance = list(zip(euclidean_distance, training_label))

        # sort distances and select first k distances
        sorted_list = sorted(distance, key=lambda x: x[0])[:k]
        knn = [label for _, label in sorted_list]

        # get most common label
        counts = Counter(knn)
        common = counts.most_common()
        max_count = common[0][1]
        predictions = [label for label, count in common if count == max_count]

        # if there is a tie, choose randomly
        if len(predictions) > 1:
            result = rd.choice(predictions)
            accuracy = 1 / len(predictions) if true_class in predictions else 0
        else:
            result = predictions[0]
            accuracy = 1 if result == true_class else 0
        total += accuracy

        # print output as specified
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % (object_id, result, true_class, accuracy))
        object_id += 1
    # compute classification accuracy and output it
    classification_accuracy = total / len(test_data)
    print('classification accuracy=%6.4f' % (classification_accuracy))      