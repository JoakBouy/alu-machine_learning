#!/usr/bin/env python3

"""This module has a function that creates
and trains a genism fastText model"""
from gensim.models import FastText

def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """creates and trains a genism fastText model
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
    sg = 0 if cbow else 1

    model = FastText(sentences=sentences,
                     vector_size=size,
                     min_count=min_count,
                     negative=negative,
                     window=window,
                     sg=sg,
                     epochs=iterations,
                     seed=seed,
                     workers=workers)
    return model
