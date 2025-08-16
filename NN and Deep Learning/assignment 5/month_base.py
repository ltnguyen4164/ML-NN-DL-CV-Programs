import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from month_solution import *

#%%

fname = "jena_climate_2009_2016.csv"
with open(fname) as f:
    data = f.read()
lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]   # The first line in the file is header information

month = np.zeros((len(lines),))
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    line_parts = line.split(",")
    values = [float(x) for x in line_parts[1:]]
    raw_data[i] = values
    date_parts = line_parts[0].split(".")
    month[i] = int(date_parts[1])
    if (month[i]==12):
        month[i] = 0

    
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

#%% Feature normalization so that each feature has mean 0 and std 1.

train_start = 0
train_end = num_train_samples

# The next line calls your data_normalization function.
normalized_data = data_normalization(raw_data, train_start, train_end)

#%% Create training, validation, test set

train_data = normalized_data[0:num_train_samples, :]
train_months = month[0:num_train_samples]

validation_start = num_train_samples
validation_end = validation_start + num_val_samples
val_data = normalized_data[validation_start:validation_end, :]
val_months = month[validation_start:validation_end]

test_start = validation_end
test_end = test_start + num_test_samples
test_data = normalized_data[test_start:test_end, :]
test_months = month[test_start:test_end]

train_size = 10000
val_size = 10000
test_size = 1000
sampling = 6

# The next lines call your make_inputs_and_targets function.

(train_inputs, train_targets) = make_inputs_and_targets(train_data, train_months, 
                                                        train_size, sampling)

(val_inputs, val_targets) = make_inputs_and_targets(val_data, val_months, 
                                                    val_size, sampling)

(test_inputs, test_targets) = make_inputs_and_targets(test_data, val_months, 
                                                      test_size, sampling)

print("\ntraining time range: (%d, %d)" % (0, num_train_samples))
print("training_inputs.shape:", train_inputs.shape)
print("training_targets.shape:", train_targets.shape)

print("\nvalidation time range: (%d, %d)" % (validation_start, validation_end))
print("validation_inputs.shape:", val_inputs.shape)
print("validation_targets.shape:", val_targets.shape)


print("\ntest time range: (%d, %d)" % (test_start, test_end))
print("test_inputs.shape:", test_inputs.shape)
print("test_targets.shape:", test_targets.shape)

# build and train a dense model

filename = "jena_dense1_16.keras"

# The next line calls your build_and_train_dense function.
history_dense = build_and_train_dense(train_inputs, train_targets, 
                                      val_inputs, val_targets, filename)

val_acc = np.array(history_dense.history["val_accuracy"])
best_val_acc = val_acc.max()
best_epoch = val_acc.argmax()+1
print("\nBest validation accuracy: %.1f%%, reached in epoch %d" % 
      (best_val_acc * 100, best_epoch))

# evaluate the dense model
# The next line calls your test_model function.
test_acc = test_model(filename, test_inputs, test_targets)

print("Dense test accuracy: %.1f%%" % (test_acc * 100))

# compute the confusion matrix for the dense model

np.set_printoptions(formatter={'int': '{: 4d}'.format})
# The next line calls your confusion_matrix function.
conf_matrix = confusion_matrix(filename, test_inputs, test_targets)

print(conf_matrix)
#print("Dense test acc = %.1f%%" % (conf_matrix.diagonal().sum() / test_size * 100))

"""
This is commented out, will be included in homework 6
# build and train a RNN model

#%%
filename = "jena_RNN_32.keras"
# The next line calls your build_and_train_rnn function.
history_rnn = build_and_train_rnn(train_inputs, train_targets, 
                                  val_inputs, val_targets, filename)

val_acc = np.array(history_rnn.history["val_accuracy"])
best_val_acc = val_acc.max()
best_epoch = val_acc.argmax()+1
print("\nBest validation accuracy: %.1f%%, reached in epoch %d" % 
      (best_val_acc * 100, best_epoch))

# evaluate the RNN model

model = keras.models.load_model(filename)
# The next line calls your test_model function.
test_acc = test_model(filename, test_inputs, test_targets)

print("RNN test accuracy: %.1f%%" % (test_acc*100))

# compute the confusion matrix for the RNN model

np.set_printoptions(formatter={'int': '{: 4d}'.format})
# The next line calls your confusion_matrix function.
conf_matrix = confusion_matrix(filename, test_inputs, test_targets)

print(conf_matrix)
#print("RNN test acc = %.1f%%" % (conf_matrix.diagonal().sum() / test_size * 100))
"""
