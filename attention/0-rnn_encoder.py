#!/usr/bin/env python3

"""
This module contains a class RNNEncoder
that encode machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class RNNEncoder"""

    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        """class contructor
        vocab - int size of input vocabulary
        embedding - dim of embedding vector
        units -  no. of hidden units in RNN cell
        batch - batch size int"""
        self.vocab = vocab
        self.embed = embedding
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=self.embed)
        self.gru = tf.keras.layers.GRU(units=units,
                                       kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        initialize hidden states"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        x - tensor - (batch, input_seq_len)
        initial - tensor-(batch, units) - intial hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
