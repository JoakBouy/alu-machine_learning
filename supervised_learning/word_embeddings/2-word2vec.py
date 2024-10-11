#!/usr/bin/env python3

"""This module has a function that
creates and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """creates and training gensim word3vec model
    sentences - list of sentences to be trained on
    size - dimensionality of embedding layer
    min_count - min no. of occurences of a word
    window - max dist btwn current and predicted word
        in a sentence
    negative - size of negative sampling
    cbow - bool to determine training type
        - True - CBOW
        - False - Skip-gram
    iterations - no. of iterations to train
    seed - seed for random no. generator
    workers - no. of worker threads to train the model
    return the trained model
    """
    # Set sg parameter based on cbow value
    sg = 0 if cbow else 1

    # create and train the word2vec model
    model = Word2Vec(sentences=sentences,
                     vector_size = size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     sg=0,
                     epochs=iterations,
                     workers=workers,
                     seed=seed)

    return model
