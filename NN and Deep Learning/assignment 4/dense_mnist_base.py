from tensorflow import keras
import numpy as np

from dense_mnist_solution import *


#%%


(training_inputs, training_labels, test_inputs, test_labels) = load_mnist()


# Creating the model

layers = 4
units_per_layer = [500,400]
epochs = 20
hidden_activations = ['tanh', 'sigmoid']


# here is where your create_and_train_model function is called
model = create_and_train_model(training_inputs, training_labels, layers, 
                               units_per_layer, epochs, hidden_activations)

#test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)
#print('\nTest accuracy: %.2f%%' % (test_acc * 100))

#
test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))

