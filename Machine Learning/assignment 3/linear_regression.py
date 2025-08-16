# Long Nguyen
# 1001705873

import numpy as np

def linear_regression(training_file, test_file, degree, lambda1):
    training_data = []
    test_data = []
    
    # check if value is float or not
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    # try to open training file and extract data
    try:
        with open(training_file) as file:
            for line in file:
                line_values = line.split()
                training_list = [float(value) if is_float(value) else value for value in line_values]
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
                line_values = line.split()
                test_list = [float(value) if is_float(value) else value for value in line_values]
                test_data.append(test_list)
    except FileNotFoundError:
        print(f"Error: The file '{test_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()

    # function to generate polynomial features based on degree
    def basis_function(x, degree):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        x_poly = np.ones((x.shape[0], 1))
        positions = [(0,)]

        for d in range(1, degree + 1):
            for i in range(x.shape[1]):
                x_poly = np.hstack((x_poly, (x[:, i] ** d).reshape(-1, 1)))
                positions.append((i,d))
        
        return x_poly, positions
    
    # function that sorts weights based on lexicographical order of the terms' positions
    def sort(w, position):
        sorted_idx = sorted(range(len(position)), key=lambda k: position[k])
        w_sorted = w[sorted_idx]
        return w_sorted
    
    # ******* TRAINING PHASE *******
    # seperate input(s) and output
    x_train = []
    y_train = []
    for i in range(0, len(training_data)):
        x_train.append([v for v in training_data[i][:-1] if isinstance(v, (int, float))])  # all but the last element (features)
        y_train.append(training_data[i][-1] if isinstance(training_data[i][-1], (int, float)) else None)  # last element (output)
    
    # convert list to np array
    x_train = np.array(x_train)
    y_train = np.array([y for y in y_train if y is not None])

    # generate polynomial features for the specified degree
    x_poly, position = basis_function(x_train, degree)
    
    # compute: 𝜆I, where I is the MxM identity matrix 
    lb = lambda1 * np.eye(x_poly.shape[1])
    
    # compute the weights
    w = np.linalg.pinv(lb + x_poly.T @ x_poly) @ x_poly.T @ y_train
         
    # sort weights
    w_sorted = sort(w, position)
    
    # print computed weights
    for i, weight in enumerate(w_sorted):
         print(f"w{i}=%.4f" % weight)

    # ******* TESTING PHASE *******
    for idx, test_obj in enumerate(test_data):
        x_test = np.array([v for v in test_obj[:-1] if isinstance(v, (int, float))])
        y_test = test_obj[-1] if isinstance(test_obj[-1], (int, float)) else None

        # generate polynomial features for the test object
        x_test_poly, temp = basis_function(x_test, degree)
        
        # predict output using learned weights
        output = np.dot(x_test_poly, w)
        
        # compute squared error
        squared_error = (output - y_test) ** 2
        
        # print output
        print(f"ID = {idx+1:5d}, output = {output[0]:14.4f}, target value = {y_test:10.4f}, squared error = {squared_error[0]:.4f}")