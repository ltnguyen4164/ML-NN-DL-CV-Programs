import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import random
import string
import re

from reverse_common import *
from reverse_solution import *
#from scratch import *

#%%

train_sentences = load_strings("reverse_dataset/reverse_train.txt")
validation_sentences = load_strings("reverse_dataset/reverse_validation.txt")
(test_sources, test_targets) = load_pairs("reverse_dataset/reverse_test.txt")
                                                      
print("read %d training sentences" % (len(train_sentences)))
print("read %d validation sentences" % (len(validation_sentences)))
print("read %d test examples" % (len(test_sources)))

#%%

# here we are calling the train_enc_dec function that you need to write.
(model, source_vec_layer, target_vec_layer) = train_enc_dec(train_sentences, 
                                                            validation_sentences,
                                                            150)

#%%
number = len(test_sources) # use this line to test on the whole test set.
#number = 100 #  use this line to test on only 100 test objects, for quicker results.
(small_test_sources, small_test_targets) = random_samples(test_sources, 
                                                          test_targets, 
                                                          number)
# here we are calling the get_enc_dec_results function that you need to write.
results = get_enc_dec_results(model, small_test_sources,
                              source_vec_layer, target_vec_layer)

wa = word_accuracy(results, small_test_targets)
print("Encoder-decoder word accuracy = %.2f%%" % (wa * 100))

#%%

# here we are calling the train_best_model function that you need to write.
(model, source_vec_layer, target_vec_layer) = train_best_model(train_sentences, 
                                                               validation_sentences)


number = len(test_sources) # use this line to test on the whole test set.
#number = 100 #  use this line to test on only 100 test objects, for quicker results.
(small_test_sources, small_test_targets) = random_samples(test_sources, 
                                                          test_targets, 
                                                          number)

# here we are calling the get_best_model_results function that you need to write.
results = get_best_model_results(model, small_test_sources,
                                 source_vec_layer, target_vec_layer)

wa = word_accuracy(results, small_test_targets)
print("Best model word accuracy = %.2f%%" % (wa * 100))



# results with 150 epochs:
#              RNN 7:      71.95%
#              other RNNs: 72.10%, 72.75%,  71.12%, 69.84%, 74.07%, 73.20%
#                          72.74%, 72.91%

# results with 50 epochs: 46.07%

# results with 75 epochs: 56.29%, 56.69%, 59.08%, 58.46%, 55.77%, 57.93%


# results with "best", from Spring 2022: 
#    96.81%, 97.71%, 95.86%, 97.25%, 96.87%
#    95.64%, 95.53%, 95.61%, 94.30%, 97.14%, 96.26%
# Training the "best" network takes 20 seconds in total, and testing it
# takes another 27 seconds.

# Spring 2023 update: results with improved "best", 10 tries: 98.28% to 99.13%
# 98.96%, 98.50%, 98.45%, 98.66%, 98.34%
# 98.55%, 99.13%, 98.50%, 98.96%, 98.28%
# Training the "best" network takes 16 seconds in total, and testing it
# takes another 15 seconds.

