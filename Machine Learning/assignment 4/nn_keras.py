# Long Nguyen
# 1001705873

import numpy as np
import random as rand
import tensorflow as tf

def parse_input(value):
    # translation dictionary for strings
    # assuming that the max number of classes is 10
    translation_dict = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.9,
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
    
def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # Make sure there are minimum 2 layers
    if layers < 2:
        print("Minimum layers is 2.")
        quit() 
    
    training_file = directory + "/" + dataset + "_training.txt"
    test_file = directory + "/" + dataset + "_test.txt"
    
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
    training_input = training_data[:, :-1]
    training_label = training_data[:, -1].reshape(-1, 1)

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
    test_input = test_data[:, :-1]
    test_label = test_data[:, -1].reshape(-1, 1)
        
    #############################TRAINING PHASE################################
    input_shape = training_input[0].shape
    num_classes =  np.max([np.max(training_label), np.max(test_label)]) + 1
        
    model = tf.keras.Sequential()
    
    # layer 1 is the input layer, so it just specifies the input shape
    model.add(tf.keras.Input(shape = input_shape))
    
    # hidden layers are layers 2 to layer L-1
    if layers > 2:
        for _ in range(layers - 2):
            model.add(tf.keras.layers.Dense(units_per_layer, activation='sigmoid'))
    
    # output layer L, where # of perceptrons = num_classes
    model.add(tf.keras.layers.Dense(int(num_classes), activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(training_input, training_label, epochs=epochs)
    
    #test_loss, test_acc = model.evaluate(test_input, test_label, verbose=0)
    #print('\nTest accuracy: %.2f%%' % (test_acc * 100))
    
    ##############################TESTING PHASE################################
    total = 0
    predict = model.predict(test_input)
    
    for i in range(len(test_input)):
        # get object id
        obj_id = i + 1
        # get true class
        tru_class = int(test_label[i, 0])
        
        # predict class
        result = predict[i]
        
        # find all tied classes using np.argwhere
        ties = np.argwhere(result == np.max(result)).flatten()
        # choose randomly between ties
        result = rand.choice(ties)
        
        # compute accuracy
        if len(ties) == 1:
            # no ties
            accuracy = 1.0 if result == tru_class else 0.0
        else:
            # has ties
            accuracy = 1.0 / len(ties) if tru_class in ties else 0.0
            
        # sum of the accuracies
        total += accuracy
        
        # print output
        print('ID=%5d, predicted=%10d, true=%10d, accuracy=%4.2f\n' % 
             (obj_id, result, tru_class, accuracy))
        
    # compute classification accuracy
    class_accuracy = total / len(test_input)
    print('Classification accuracy=%6.4f\n' % class_accuracy)