# Long Nguyen
# 1001705873

import numpy as np
import tensorflow as tf
from tensorflow import keras

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                         key_dim=embed_dim)
        self.dense_proj = keras.Sequential([keras.layers.Dense(dense_dim, activation="relu"),
                                            keras.layers.Dense(embed_dim),])
        
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
    
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,
                       "num_heads": self.num_heads,
                       "dense_dim": self.dense_dim})
        return config
    
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(input_dim=input_dim,
                                                       output_dim=output_dim)
        self.position_embeddings = keras.layers.Embedding(input_dim=sequence_length,
                                                          output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        
        return embedded_tokens + embedded_positions
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim,
                       "sequence_length": self.sequence_length,
                       "input_dim": self.input_dim})
        return config

def train_transformer(train_inputs, train_labels, validation_inputs, validation_labels):
    # define parameters
    vocab_size = 250  
    sequence_length = 20 
    embed_dim = 100
    num_heads = 3
    dense_dim = 32
    epochs = 10

    # create vectorization layer
    vect_layer = keras.layers.TextVectorization(max_tokens=vocab_size,
                                                output_mode="int",
                                                output_sequence_length=sequence_length)

    # adapt the layer using training data
    train_ds = tf.data.Dataset.from_tensor_slices(train_inputs).batch(32)
    vect_layer.adapt(train_ds)

    # create the model with transformer and positional embedding
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = vect_layer(inputs)
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)

    # compile and train the model
    train_inputs = np.array(train_inputs, dtype=object)
    validation_inputs = np.array(validation_inputs, dtype=object)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_inputs, train_labels, validation_data=(validation_inputs, validation_labels), epochs=epochs, batch_size=32)

    return model, vect_layer
def evaluate_transformer(model, text_vectorization, test_inputs, test_labels):
    # convert test inputs to proper format
    test_inputs = np.array(test_inputs, dtype=object)
    
    # evaluate model
    loss, accuracy = model.evaluate(test_inputs, test_labels, batch_size=32, verbose=0)

    return accuracy