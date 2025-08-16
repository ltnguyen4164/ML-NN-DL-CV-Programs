#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import random
import string
import re


def save_pairs(filename, pairs):
    f = open(filename, "w")
    
    for pair in pairs:
        print(pair[0]+"\t"+pair[1], file=f)
    f.close()    


def load_pairs(filename):
    f = open(filename)
    lines = f.readlines()
    number = len(lines)
    sources = [None] * number
    targets = [None] * number
    counter = 0
    for line in lines:
        line = line.strip()
        pair = line.split('\t')
        if (len(pair) != 2):
            print("failed to parse this line:\n%s" & (line))
            print(pair)
        sources[counter] = pair[0]
        targets[counter] = pair[1]
        counter = counter + 1
    return (sources, targets)

def save_strings(filename, strings):
    f = open(filename, "w")
    for s in strings:
        print(s, file=f)
    f.close()
    
def load_strings(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    result = [line.strip() for line in lines]
    return result


def random_samples(sources, targets, number):
    indices = list(range(0, len(sources)))
    random.shuffle(indices)
    result_sources = [sources[i] for i in indices[0:number]]
    result_targets = [targets[i] for i in indices[0:number]]
    
    return(result_sources, result_targets)


def standardize(strings):
    tv = layers.TextVectorization(output_mode="int")
    tv.adapt(strings)
    vocab = tv.get_vocabulary()
    vectors = tv(strings).numpy()
    
    result = []
    for vector in vectors:
        ints = vector[vector > 0]
        words = [vocab[i] for i in ints]
        text = " ".join(words)
        result = result + [text]
    
    return result


def count_matches(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    n1 = len(w1)
    n2 = len(w2)
    n_min = min(n1, n2)
    n_max = max(n1, n2)
    
    count = 0
    for i in range(0, n_min):
        if (w1[i] == w2[i]):
            count += 1
    
    return (count, n_max)


def word_accuracy(results, targets):
    counts = 0
    totals = 0
    num = len(results)
    targets = standardize(targets)
    
    for i in range(0, num):
        (count, total) = count_matches(results[i], targets[i])
        counts += count
        totals += total
    
    return counts / totals


