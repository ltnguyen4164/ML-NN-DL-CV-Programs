from tensorflow import keras
import numpy as np

from cnn_mnist_solution import load_mnist, create_and_train_model


#%%


(training_inputs, training_labels, test_inputs, test_labels) = load_mnist()


#%%
# Creating the model

blocks = 2
filter_size = 3
filter_number = 32
region_size = 2
epochs = 20
cnn_activation = 'relu'

# here is where your create_and_train_model function is called
model = create_and_train_model(training_inputs, training_labels, blocks, 
                               filter_size, filter_number, region_size, 
                               epochs, cnn_activation)

#
test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))

#%%
