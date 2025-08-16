import numpy as np
import os
import sys


# (layers, units, biases, weights) = nn_load(pathname)
# pathname: specifies the name of a local file
# layers: an integer, specifies the number of layers in the network.
# units: a list of integers specifying the number of units in each layer.
    # units[L] is the number of units in layer L (where layer 1 is the input layer).
    # units[0] = None, since in our notation there is no layer 0.
    # units[1] is the number of units in the input layer
    # units[layers] is the number of units in the output layer
# biases: a list of numpy column matrices. 
    # biases[L] contains the biases in layer L (where layer 1 is the input layer).
    # biases[0] = None, since in our notation there is no layer 0.
    # biases[1] = None, since the input layer contains no perceptrons and thus no bias values
    # biases[2] contains the biases in the first hidden layer
    # biases[layers] contains the biases in the output layer.
# weights: a list of numpy matrices specifying the weights in each layer.
    # weights[L] contains the weights in layer L (where layer 1 is the input layer).
        # Its number of rows is the number of units in layer L (saved as units[L])
        # Its number of columns is the number of units in layer L-1 (saved as units[L-1])
        # Every row specifies the weights in a unit of layer L
    # weights[0] = None, since in our notation there is no layer 0.
    # weights[1] = None, since the input layer contains no perceptrons and thus no weight values
    # weights[2] contains the weights in the first hidden layer
    # weights[layers-2] contains the biases in the output layer.
# With the exception of the input layer, each layer L is assumed to be fully 
# connected, meaning that the input of each unit in layer L is connected
# to the outputs of all units in layer L-1.

def nn_load(pathname):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    
    # get number of layers
    line = in_file.readline()
    items = line.split()
    if (len(items) != 2) or (items[0] != "layers:"):
        print("nn_load: invalid layers line %s" %(line))
        in_file.close()
        return None
    
    layers = int(items[1])

    # get number of units in each layer
    units = (layers+1) * [None]
    line = in_file.readline()
    items = line.split()
    if (len(items) != layers+1) or (items[0] != "units:"):
        print("nn_load: invalid units line %s" %(line))
        in_file.close()
        return None
    
    for i in range(0, layers):
        units[i+1] = int(items[i+1])
    
    # read the biases and weights for each layer
    biases = [None, None]
    weights = [None, None]
    
    for i in range(2, layers+1):
        current_units = units[i]
        previous_units = units[i-1]
        
        line = in_file.readline().strip()
        if (line != "start layer"):
            print("nn_load: expected 'start layer', invalid line %s" %(line))
            in_file.close()
            return None

        # read current bias
        line = in_file.readline().strip()
        if (line != "start bias"):
            print("nn_load: expected 'start bias', invalid line %s" %(line))
            in_file.close()
            return None
        b = read_2D_array(in_file, 1, current_units)
        b = b.transpose();
        biases.append(b)
        line = in_file.readline().strip()
        if (line != "end bias"):
            print("nn_load: expected 'end bias', invalid line %s" %(line))
            in_file.close()
            return None
    
        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "start w"):
            print("nn_load: expected 'start w', invalid line %s" %(line))
            in_file.close()
            return None
        w = read_2D_array(in_file, current_units, previous_units)
        weights.append(w)
        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "end w"):
            print("nn_load: expected 'start w', invalid line %s" %(line))
            in_file.close()
            return None
        
    in_file.close()
    return (layers, units, biases, weights)
    

# arguments:
#    in_file: source file of the data. This is NOT a pathname, it is a 
#       python file object that has already been opened.
#    rows, cols: size of the matrix that should be read 
# returns:
#    result: a matrix of size rows x cols
def read_2D_array(in_file, rows, cols):
    result = np.zeros((rows, cols))
    
    for row in range(0, rows):
        file_line = in_file.readline()
        items = file_line.split()
        cols2 = len(items)
        if (cols2 != cols):
            print("read_matrix: Line %d, %d columns expected, %d columns found" %(row, cols, cols2))
            return None
        for col in range(0, cols):
            result[row][col] = float(items[col])

    return result
    

# returns a 2D matrix from a file stored at the specified pathname.
def read_matrix(pathname):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    file_lines = in_file.readlines()
    in_file.close()

    rows = len(file_lines)
           
    in_file = open(pathname)
    data = read_2D_array(in_file, rows, 1)
    in_file.close()
    return data


def print_nn_info(layers, units, biases, weights):
    print("There are %d layers" % (layers))
    print("Units in each layer:", units[1:])
    for i in range(2, len(biases)):
        print("layer %d biases:" % (i), biases[i][:,0])
    print()
    for i in range(2, len(weights)):
        print("layer %d weights:" % (i))
        print(weights[i])


def test_nn_load():
    default_directory = "."
    default_nn_file = "nn_xor.txt"
    
    if (len(sys.argv) >= 2):
        nn_file = sys.argv[1]
    else:
        nn_file = default_directory + "/" + default_nn_file
    
    (layers, units, biases, weights) = nn_load(nn_file)
    print_nn_info(layers, units, biases, weights)

#test_nn_load()
