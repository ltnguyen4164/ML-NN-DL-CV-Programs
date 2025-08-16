import numpy as np
import tensorflow as tf
from tensorflow import keras

from authors_solution import *


train_files = [["authors_dataset/train_data/dickens_david_copperfield.txt"],
               ["authors_dataset/train_data/lewis_narnia_caspian.txt",
                "authors_dataset/train_data/lewis_narnia_lww.txt"],
               ["authors_dataset/train_data/twain_tom_sawyer.txt"],]

model = learn_model(train_files)

batch_size = 32
test_ds = keras.utils.text_dataset_from_directory("authors_dataset/test_tf", 
                                                  batch_size=batch_size)

(test_loss, test_acc) = model.evaluate(test_ds)

print("Test accuracy: %.2f%%" % (test_acc*100))
