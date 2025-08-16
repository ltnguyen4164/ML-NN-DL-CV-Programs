#%%

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from cifar2mnist_solution import *  # this is the file that you need to create.

(training_set, test_set) = keras.datasets.cifar10.load_data()

(cifar_tr_inputs, cifar_tr_labels) = training_set
(cifar_test_inputs, cifar_test_labels) = test_set



num_classes = cifar_tr_labels.max()+1

input_shape = cifar_tr_inputs[0].shape
model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="tanh"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="tanh"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="tanh"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="tanh"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

model.summary()

batch_size = 128
epochs = 20

# You need to write this function
train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs)

pretrained_model_file = 'cifar10_e20_b128_6.keras'
model.save(pretrained_model_file)

#%%

dataset = keras.datasets.mnist

# the data, split between train and test sets
(mnist_tr_inputs, mnist_tr_labels), (mnist_test_inputs, mnist_test_labels) = dataset.load_data()
number_of_classes = np.max([np.max(mnist_tr_labels), np.max(mnist_test_labels)]) + 1

# Scale images to the [0, 1] range
mnist_tr_inputs = mnist_tr_inputs.astype("float32") / 255
mnist_test_inputs = mnist_test_inputs.astype("float32") / 255

class_indices = [None] * num_classes
for c in range(0, num_classes):
    (indices) = np.nonzero(mnist_tr_labels == c)
    class_indices[c] = indices[0]

samples_per_class = 100
small_indices = class_indices[0][0:samples_per_class]
for c in range(1, num_classes):
    small_indices = np.concatenate((small_indices, class_indices[c][0:samples_per_class]))
    
small_tr_inputs = mnist_tr_inputs[small_indices]
small_tr_labels = mnist_tr_labels[small_indices]

print("small_tr_inputs shape:", small_tr_inputs.shape)
print("mnist_test_inputs shape:", mnist_test_inputs.shape)

#%%

epochs = 40
batch_size = 32

# You need to write this function
refined_model = load_and_refine(pretrained_model_file, small_tr_inputs, 
                                small_tr_labels, batch_size, epochs)

refined_model_file = "refined_mnist100_e40_b32.keras"
refined_model.save(refined_model_file)



refined_model = keras.models.load_model('refined_mnist100_e40_b32.keras')

# You need to write this function
test_acc = evaluate_my_model(refined_model, mnist_test_inputs, mnist_test_labels)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))

