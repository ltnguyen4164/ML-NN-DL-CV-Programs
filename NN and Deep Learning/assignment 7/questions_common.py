import numpy as np


def load_inputs(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    result = [line.strip() for line in lines]
    return result


def load_labels(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    ints = [int(line) for line in lines]
    result = np.array(ints)
    return result


# This code prints out all the mistakes in the test set.
def print_mistakes(model, text_vectorization, test_inputs, test_labels):
    test_vectors = text_vectorization(test_inputs).numpy()
    results = model(test_vectors).numpy()[:,0]
    results[results < 0.5] = 0
    results[results >= 0.5] = 1
    
    mistakes = (results-test_labels).nonzero()
    print("mistake indices: ", mistakes)
    for m in mistakes[0]:
        print("index: %4d. " % (m), "label: %d. " % (test_labels[m]), test_inputs[m])

