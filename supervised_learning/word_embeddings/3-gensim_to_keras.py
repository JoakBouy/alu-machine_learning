#!/usr/bin/env python3

"""This module has a function that converts
gensim word2vec model to a keras embedding layer"""
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """convert word2vec model to keras layer
    model - trained gensim wordvec models
    returns trainable keras Embedding
    """
    # Get the vocab size and embedding dimension from the Word2Vec model
    vocab_size, embedding_dim = model.wv.vectors.shape

    # Initialize an empty embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Populate the embedding matrix with the Word2Vec embeddings
    for word, i in model.wv.key_to_index.items():
        embedding_matrix[i] = model.wv[word]

    # Create a Keras Embedding layer and set its weights
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
        )

    return embedding_layer