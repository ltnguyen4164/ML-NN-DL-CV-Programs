# Long Nguyen
# 1001705873

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import string
import re

from transformers_common import *

def train_enc_dec(train_sentences, validation_sentences, epochs):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    
    def custom_standardization(input):
        lowercase = tf.strings.lower(input)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")
    
    # define text vectorization layers
    vocab_size = 300
    src_vectorization = keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int")
    target_vectorization = keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int", standardize=custom_standardization)
    
    # create source and target sentence pairs
    start_token = "[start]"
    end_token = "[end]"
    def shuffle_sentence(sentence):
        words = sentence.strip().split()
        if random.random() < 0.5:
            words = words[::-1]
        return " ".join(words)

    train_pairs = [(shuffle_sentence(s), f"{start_token} {s} {end_token}") for s in train_sentences]
    val_pairs = [(shuffle_sentence(s), f"{start_token} {s} {end_token}") for s in validation_sentences]
    
    # compute vocabulary
    train_src, train_tgt = zip(*train_pairs)
    src_vectorization.adapt(list(train_src))
    target_vectorization.adapt(list(train_tgt))
    
    # function that creates input and target sequences
    def format_dataset(input, target):
        input = src_vectorization(input)
        target = target_vectorization(target)
        return ({"encoder": input,
                 "decoder": target[:, :-1]},
                 target[:, 1:])
    # function that creates the actual optimized Tensorflow datasets
    def make_dataset(pairs):
        in_texts, out_texts = zip(*pairs)
        in_texts = list(in_texts)
        out_texts = list(out_texts)
        dataset = tf.data.Dataset.from_tensor_slices((in_texts, out_texts))
        dataset = dataset.batch(batch_size=64)
        dataset = dataset.map(format_dataset)
        return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()
    
    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    # make encoder & decoder model
    embed_dim = 64
    latent_dim = 64

    src = keras.Input(shape=(None,), dtype="int64", name="encoder")
    x1 = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(src)
    encoded_src = keras.layers.Bidirectional(keras.layers.GRU(latent_dim), merge_mode="sum")(x1)

    past_target = keras.Input(shape=(None,), dtype="int64", name="decoder")
    x2 = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
    decoder_gru = keras.layers.GRU(latent_dim, return_sequences=True)
    x3 = decoder_gru(x2, initial_state=encoded_src)
    x4 = keras.layers.Dropout(0.5)(x3)
    target_next_step = keras.layers.Dense(vocab_size, activation="softmax")(x4)
    
    seq2seq_rnn = keras.Model([src, past_target], target_next_step)

    # train the network
    seq2seq_rnn.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    seq2seq_rnn.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=64)
    
    return seq2seq_rnn, src_vectorization, target_vectorization

def get_enc_dec_results(model, test_sentences, source_vec_layer, target_vec_layer):
    vocab = target_vec_layer.get_vocabulary()
    index_to_word = dict(enumerate(vocab))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    start_token = "[start]"
    end_token = "[end]"
    start_token_idx = word_to_index[start_token]
    end_token_idx = word_to_index[end_token]

    max_length = 20
    batch_size = 64
    results = []

    for i in range(0, len(test_sentences), batch_size):
        batch = test_sentences[i:i+batch_size]
        batch_size_actual = len(batch)

        encoder_input = source_vec_layer(batch)
        decoder_input = tf.fill([batch_size_actual, 1], start_token_idx)
        finished = tf.zeros([batch_size_actual], dtype=tf.bool)

        for _ in range(max_length):
            predictions = model.predict([encoder_input, decoder_input], verbose=0)
            next_token_logits = predictions[:, -1, :]
            next_token_ids = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)

            decoder_input = tf.concat([decoder_input, tf.expand_dims(next_token_ids, axis=1)], axis=1)
            finished = tf.logical_or(finished, tf.equal(next_token_ids, end_token_idx))
            if tf.reduce_all(finished):
                break

        # convert token sequences to strings
        for sequence in decoder_input.numpy():
            words = [index_to_word[token] for token in sequence if token != 0]
            if end_token in words:
                words = words[1:words.index(end_token)]
            else:
                words = words[1:]
            results.append(" ".join(words))

    return results

def train_best_model(train_sentences, validation_sentences):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    
    def custom_standardization(input):
        lowercase = tf.strings.lower(input)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")
    
    # define text vectorization layers
    vocab_size = 300
    src_vectorization = keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int")
    target_vectorization = keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int", standardize=custom_standardization)
    
    # create source and target sentence pairs
    start_token = "[start]"
    end_token = "[end]"
    def shuffle_sentence(sentence):
        words = sentence.strip().split()
        if random.random() < 0.5:
            words = words[::-1]
        return " ".join(words)

    train_pairs = [(shuffle_sentence(s), f"{start_token} {s} {end_token}") for s in train_sentences]
    val_pairs = [(shuffle_sentence(s), f"{start_token} {s} {end_token}") for s in validation_sentences]
    
    # compute vocabulary
    train_src, train_tgt = zip(*train_pairs)
    src_vectorization.adapt(list(train_src))
    target_vectorization.adapt(list(train_tgt))
    
    # function that creates input and target sequences
    def format_dataset(input, target):
        input = src_vectorization(input)
        target = target_vectorization(target)
        return ({"encoder": input,
                 "decoder": target[:, :-1]},
                 target[:, 1:])
    # function that creates the actual optimized Tensorflow datasets
    def make_dataset(pairs):
        in_texts, out_texts = zip(*pairs)
        in_texts = list(in_texts)
        out_texts = list(out_texts)
        dataset = tf.data.Dataset.from_tensor_slices((in_texts, out_texts))
        dataset = dataset.batch(batch_size=64)
        dataset = dataset.map(format_dataset)
        return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()
    
    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    embed_dim = 64
    sequence_length = 20
    num_heads = 3
    dense_dim = 32

    # create model
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = keras.layers.Dropout(0.5)(x)

    decoder_outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # train model
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=30)

    return model, src_vectorization, target_vectorization

def get_best_model_results(model, test_sentences, source_vec_layer, target_vec_layer):
    vocab = target_vec_layer.get_vocabulary()
    index_to_word = dict(enumerate(vocab))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    start_token = "[start]"
    end_token = "[end]"
    start_token_idx = word_to_index[start_token]
    end_token_idx = word_to_index[end_token]

    max_length = 20
    batch_size = 64
    results = []

    for i in range(0, len(test_sentences), batch_size):
        batch = test_sentences[i:i+batch_size]
        batch_size_actual = len(batch)

        encoder_input = source_vec_layer(batch)
        decoder_input = tf.fill([batch_size_actual, 1], start_token_idx)
        finished = tf.zeros([batch_size_actual], dtype=tf.bool)

        for _ in range(max_length):
            predictions = model.predict([encoder_input, decoder_input], verbose=0)
            next_token_logits = predictions[:, -1, :]
            next_token_ids = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)

            decoder_input = tf.concat([decoder_input, tf.expand_dims(next_token_ids, axis=1)], axis=1)
            finished = tf.logical_or(finished, tf.equal(next_token_ids, end_token_idx))
            if tf.reduce_all(finished):
                break

        # Convert token sequences to strings
        for sequence in decoder_input.numpy():
            words = [index_to_word[token] for token in sequence if token != 0]
            if end_token in words:
                words = words[1:words.index(end_token)]
            else:
                words = words[1:]
            results.append(" ".join(words))

    return results