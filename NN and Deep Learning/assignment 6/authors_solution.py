# Long Nguyen
# 1001705873

import numpy as np
import tensorflow as tf
import random

def learn_model(train_files):
    def split_text_into_chunks(text, chunk_size=300, max_chunks=1000):
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks[:max_chunks]
    
    # Load texts and labels
    texts = []
    labels = []

    for label, file_list in enumerate(train_files):
        author_chunks = []
        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='cp1252') as f:
                    text = f.read()
            chunks = split_text_into_chunks(text, chunk_size=300)
            author_chunks.extend(chunks)
        
        # Shuffle and select up to 1000 chunks per author
        random.shuffle(author_chunks)
        author_chunks = author_chunks[:1000]
        texts.extend(author_chunks)
        labels.extend([label] * len(author_chunks))

    # Convert texts and labels into a tf dataset
    train_ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    train_ds = train_ds.shuffle(buffer_size=len(texts), seed=42)

    batch_size = 32
    train_ds = train_ds.batch(batch_size)

    # Text preprocessing
    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=20000,
        ngrams=2,
        output_mode='multi_hot'
    )

    # Adapt text vectorizer
    text_only = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only)

    # Initialize and compile model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        text_vectorization,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(train_files), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_ds.cache().prefetch(tf.data.AUTOTUNE), epochs=10)
    return model