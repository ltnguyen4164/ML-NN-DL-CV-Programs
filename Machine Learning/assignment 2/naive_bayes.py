# Long Nguyen
# 1001705873

import numpy as np
import math
import random

def naive_bayes(training_file, test_file):
    training_data = []
    test_data = []
    
    # try to open training file and extract data
    try:
        with open(training_file) as file:
            for line in file:
                training_list = [float(value) for value in line.split()]
                training_data.append(training_list)
    except FileNotFoundError:
        print(f"Error: The file '{training_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()

    # try to open testing file and extract data
    try:
        with open(test_file) as file:
            for line in file:
                test_list = [float(value) for value in line.split()]
                test_data.append(test_list)
    except FileNotFoundError:
        print(f"Error: The file '{test_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()

    # Organize the training data by class
    for i in range(0, len(training_data)):
        swapped = False

        for j in range(0, len(training_data)-i-1):
            if training_data[j][len(training_data[j])-1] > training_data[j+1][len(training_data[j])-1]:
                training_data[j], training_data[j+1] = training_data[j+1], training_data[j]
                swapped = True
        if swapped == False:
            break

    # Variable to hold the number of each training data with class C
    num_of_class = {}
    
    # Calculate mean and standard deviation for each attribute/column of a class
    mean = 0
    var = 0
    stdev = 0

    # Variable to keep track of the number of objects/rows in each attribute
    counter = 0

    # Save the calculated data
    training_sum = {}
    
    # Holds the accuracies of each classification
    class_acc = []
    
    # i represents the classes in the training data
    # j represents the attributes in each class
    # k represents the rows in each class
    for i in range(0, int(training_data[len(training_data)-1][len(training_data[0])-1])):
        # Create an entry for each class
        training_sum[i + 1] = []
        num_of_class[i + 1] = []

        for j in range(0, len(training_data[i])-1):
            # Reset mean and variance
            mean = 0
            var = 0
            counter = 0

            # Calculate the mean of each attribute in class i
            for k in range(0, len(training_data)):
                if int(training_data[k][len(training_data[j])-1]) != i+1:
                    continue
                else:
                    mean += training_data[k][j]
                    counter += 1
            if counter != 0:
                mean = mean / counter
            else:
                mean = 0    
            
            # Calculate the variance of each attribute in class i
            for k in range(0, len(training_data)):
                if int(training_data[k][len(training_data[j])-1]) != i+1:
                    continue
                else:
                    diff = mean - training_data[k][j]
                    sqr = pow(diff, 2)
                    var += sqr
            if counter != 0:
                var = var / (counter - 1) 
            else:
                var = 0
            # Make sure that the variance isn't a value below 0.0001       
            if var < 0.0001:
                var = 0.0001
            
            # Calculate the standard deviation of each attribute in class i
            stdev = math.sqrt(var)
            
            # Store the mean and standard deviation for this class and attribute
            training_sum[i + 1].append((mean, stdev))
            
            print("Class: %d, Attribute: %d, mean = %.2f, std = %.2f" % (i+1, j+1, mean, stdev))

        # Store the number of training data with class i
        num_of_class[i + 1] = counter
    '''
    # Calculate P(C), the percentage of training examples whose class label is C
    P_C = {}
    for class_lb, num in num_of_class.items():
        P_C[class_lb] = num / len(training_data)
    
    # i represents the ID
    for i in range(0, len(test_data)):
        # Calculate P(x | C), using the Gaussian formula and product rule
        P_x_C = {}
        # j represents the class
        for j, summary in training_sum.items():
            # Product rule variable
            p_prob = 1
            # k represents the attribute
            for k in range(0, len(training_sum[j])):
                mean, stdev = summary[k]
                x = test_data[i][k]
                prob = gaussian_probability(x, mean, stdev)
                p_prob *= prob 
            P_x_C[j] = p_prob

        # Calculate P(x), using the sum rule
        P_x = 0
        # Assumes that P_x_C and P_C are the same length since the length should be equal to the number of classes
        for j in range(0, len(P_x_C)):
            P_x += P_x_C[j+1] * P_C[j+1]
        
        # Calculate P(C | x), using Bayes Rule       
        P_C_x = {}
        for j in range(0, len(P_x_C)):
            if P_x == 0:
                P_C_x[j] = 0
            else:
                P_C_x[j] = (P_x_C[j+1] * P_C[j+1]) / P_x

        # Using P(C | x) value, predict the class
        predictions = []
        mx = max(P_C_x.values())
        for j, prob in P_C_x.items():
            if prob == mx:
                predictions.append(j)
        # If there is a tie among two or more classes, choose one of them randomly
        if len(predictions) > 1:
            result = int(random.choice(predictions))
        else:
            result = int(predictions[0])
        # Check the accuracy
        if len(predictions) == 1:
            if (result+1) == int(test_data[i][-1]):
                accuracy = 1
            else:
                accuracy = 0
        else:
            if (test_data[i][-1] - 1) in predictions:
                accuracy = 1 / len(predictions)
            else:
                accuracy = 0
        class_acc.append(accuracy)
        # Just learned that using -1 in a 2d array gives the last column
        print("ID = %5d, predicted = %3s, probability = %.4f, true = %3d, accuracy = %4.2f" % (i+1, result+1, P_C_x[result], test_data[i][-1], accuracy))
    print("classification accuracy = %6.4f" % (sum(class_acc) / len(class_acc)))
'''
# Function that calculates the gaussian probability
def gaussian_probability(x, mean, stdev):
    e = math.exp(-((x - mean) ** 2) / (2 * (stdev ** 2)))
    return (1 / (math.sqrt(2 * np.pi) * stdev)) * e   