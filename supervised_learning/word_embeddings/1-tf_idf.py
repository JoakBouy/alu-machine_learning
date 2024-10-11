#!/usr/bin/env python3

"""This module has a function that
creates a TF-IDF embedding"""
import re
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding

    parameters:
        sentences [list]:
            list of sentences to analyze

        vocab [list]:
            list of vocabulary words to use for analysis
            if None, all words within sentences should be used

    returns:
        embeddings,features:
            embeddings [numpy.ndarray of shape (s, f)]:
                contains the embeddings
                s: number of sentences in sentences
                f: number of features analyzed
            features [list]:
                list of features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embeddings = x.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features


# def compute_idf(sentences):
#     """Compute inverse document frequency for a collection of sentences"""
#     N = len(sentences)
#     idf = {}
#     all_words = set(word for doc in sentences for word in doc)

#     for word in all_words:
#         containing_docs = sum(1 for doc in sentences if word in doc)
#         idf[word] = math.log(N / (1 + containing_docs))

#     return idf


# def compute_tf(sentence):
#     """Compute term frequency for a single document"""
#     word_counts = Counter(sentence)
#     total_words = len(sentence)
#     tf = {word: count / total_words for word, count in word_counts.items()}
#     return tf


# def tf_idf(sentences, vocab=None):
#     """creates a bag of words embedding matrix
#     sentences - list of sentences to analyze
#     vocab - list of vocab words for analysis
#     returns embeddings, features
#     embeddings shape - (s, f)
#         s-no. of sentences in sentences
#         f-mo. of features analyzed
#     features - list of features for embeddings
#     """

    # tokenize sentences
    # tokenized_sentences = []
    # for sentence in sentences:
    #     tokenized_sentences.append(re.findall(r'\b\w+\b',
    #                                           sentence.lower()))
    # # print(tokenized_sentences)

    # # if vocab is not provided
    # if vocab is None:
    #     vocab_set = set()
    #     for sentence in tokenized_sentences:
    #         for word in sentence:
    #             vocab_set.add(word)
    #     vocab = sorted(vocab_set)

    # # word index dict for vocabs
    # word_index = {}
    # for i, word in enumerate(vocab):
    #     word_index[word] = i

    # # calculate idf
    # idf = compute_idf(tokenized_sentences)

    # # initialize TF-IDF matrix with 0s
    # tf_idf_matrix = np.zeros((len(sentences), len(vocab)),
    #                          dtype=float)

    # # fill TF_IDF matrix
    # for i, sentence in enumerate(tokenized_sentences):
    #     tf = compute_tf(sentence)
    #     for word, tf_value in tf.items():
    #         if word in word_index:
    #             tf_idf_matrix[i,
    #                           word_index[word]] = tf_value * idf[word]

    # return tf_idf_matrix, vocab
